#!/bin/bash
set -e

MODE="${1:-}"

if [ "$MODE" != "tp1" ] && [ "$MODE" != "tp2" ]; then
    echo "Usage: $0 {tp1|tp2}"
    exit 1
fi

REAL_USER="${LOGNAME:-$(id -un)}"
export MLPERF_ROOT="${MLPERF_ROOT:-/scratch/${REAL_USER}/mlperf}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_ROOT="$MLPERF_ROOT/training/small_llm_pretraining/nemo"

if [ "$MODE" = "tp1" ]; then
    CONFIG="$REPO_ROOT/configs/h200/h200_2gpu_tp1_dp2.sh"
else
    CONFIG="$REPO_ROOT/configs/h200/h200_2gpu_tp2_dp1.sh"
fi

source "$CONFIG"

CONTAINER="$MLPERF_ROOT/containers/mlperf-llama31-h200.sif"

PYTHONFIX_PATH="$REPO_ROOT/patches/pythonfix"

test -f "$PYTHONFIX_PATH/pangu.py" || {
    echo "ERROR: Python compatibility fix not found:"
    echo "       $PYTHONFIX_PATH/pangu.py"
    exit 1
}

mkdir -p \
    "$JOB_DIR" \
    "$TMP_NPY_INDEX" \
    "$CONTINUAL_CKPT"

test -f "$CONTAINER" || {
    echo "ERROR: container not found: $CONTAINER"
    exit 1
}

test -d "$PREPROCESSED_PATH" || {
    echo "ERROR: preprocessed C4 not found: $PREPROCESSED_PATH"
    exit 1
}

test -d "$TOKENIZER_PATH" || {
    echo "ERROR: tokenizer/model not found: $TOKENIZER_PATH"
    exit 1
}

test -f "$TRAIN_ROOT/pretrain_llama31_kisti.py" || {
    echo "ERROR: KISTI training source not installed in $TRAIN_ROOT"
    exit 1
}

RUNLOG="$JOB_DIR/launcher_$(date +%Y%m%d_%H%M%S).log"

echo "===== MLPerf Training Launch ====="
echo "Mode             : $MODE"
echo "MLPERF_ROOT      : $MLPERF_ROOT"
echo "Training source  : $TRAIN_ROOT"
echo "Container        : $CONTAINER"
echo "TP               : $TENSOR_PARALLEL_SIZE"
echo "GPUs             : $GPUS_PER_NODE"
echo "MBS / GBS        : $MBS / $GBS"
echo "Job directory    : $JOB_DIR"
echo "Launcher log     : $RUNLOG"

nohup singularity exec --nv \
    -B /scratch:/scratch \
    -B /home01:/home01 \
    -B "$JOB_DIR:/output" \
    -B "$JOB_DIR:/outputs" \
    -B "$JOB_DIR:/mlperf-outputs" \
    -B "$PREPROCESSED_PATH:/preproc_data" \
    -B "$TOKENIZER_PATH:/tokenizer" \
    -B "$TMP_NPY_INDEX:/npy_index" \
    "$CONTAINER" \
    /bin/bash -lc "
        export PYTHONPATH=$REPO_ROOT/patches/pythonfix:\$PYTHONPATH
        export TRITON_LIBCUDA_PATH=/.singularity.d/libs
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        cd $TRAIN_ROOT

        source $CONFIG

        /bin/bash ./run_llama31_kisti.sh
    " > "$RUNLOG" 2>&1 &

PID=$!

echo "PID=$PID"
echo "LOG=$RUNLOG"
