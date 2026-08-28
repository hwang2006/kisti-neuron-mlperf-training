#!/bin/bash
set -euo pipefail

MODE="${1:-tp4_dp2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${MLPERF_ROOT:?MLPERF_ROOT must be set}"

case "$MODE" in
    tp4_dp2)
        CONFIG="$REPO_ROOT/configs/a100/a100_8gpu_tp4_dp2.sh"
        ;;
    *)
        echo "Unknown A100 configuration: $MODE" >&2
        echo "Supported: tp4_dp2" >&2
        exit 1
        ;;
esac

NEMO_DIR="$MLPERF_ROOT/training/small_llm_pretraining/nemo"
export IMAGE="$MLPERF_ROOT/containers/mlperf-llama31-h200.sif"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Singularity image not found: $IMAGE" >&2
    exit 1
fi

if [ ! -f "$NEMO_DIR/run_llama31_a100.sh" ]; then
    echo "ERROR: A100 runtime source is not installed:" >&2
    echo "  $NEMO_DIR/run_llama31_a100.sh" >&2
    echo "Run scripts/02_prepare_mlcommons_source.sh first." >&2
    exit 1
fi

source "$CONFIG"

mkdir -p \
    "$JOB_DIR" \
    "$TMP_NPY_INDEX" \
    "$CONTINUAL_CKPT"

echo "============================================================"
echo " KISTI NEURON MLPerf Training"
echo " NVIDIA A100 x ${GPUS_PER_NODE}"
echo " TP=${TENSOR_PARALLEL_SIZE}"
echo " DP=$((GPUS_PER_NODE / TENSOR_PARALLEL_SIZE))"
echo " GBS=${GBS}"
echo " MBS=${MBS}"
echo " DISABLE_FP8=${DISABLE_FP8}"
echo " ENABLE_RECOMPUTE=${ENABLE_RECOMPUTE}"
echo " DISABLE_CE_FUSION=${DISABLE_CE_FUSION}"
echo " JOB_DIR=${JOB_DIR}"
echo "============================================================"

RUN_LOG="$JOB_DIR/launcher.log"

nohup singularity exec --nv \
    --bind "$MLPERF_ROOT:$MLPERF_ROOT" \
    --bind "$PREPROCESSED_PATH:/preproc_data" \
    --bind "$TOKENIZER_PATH:/tokenizer" \
    --bind "$TMP_NPY_INDEX:/npy_index" \
    --bind "$JOB_DIR:/outputs" \
    --bind "$JOB_DIR:/mlperf-outputs" \
    "$IMAGE" \
    /bin/bash -c "
        export PYTHONPATH=$REPO_ROOT/patches/pythonfix:\$PYTHONPATH
        export TRITON_LIBCUDA_PATH=/.singularity.d/libs
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        cd $NEMO_DIR
        source $CONFIG

        /bin/bash run_llama31_a100.sh
    " > "$RUN_LOG" 2>&1 &

PID=$!

echo "A100 training launched."
echo "PID=$PID"
echo "LOG=$RUN_LOG"
echo
echo "Monitor with:"
echo "  tail -f $RUN_LOG"
