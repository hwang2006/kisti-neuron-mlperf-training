#!/bin/bash
set -e

REAL_USER="${LOGNAME:-$(id -un)}"
export MLPERF_ROOT="${MLPERF_ROOT:-/scratch/${REAL_USER}/mlperf}"

CONTAINER="$MLPERF_ROOT/containers/mlperf-llama31-h200.sif"

MERGED_C4_PATH="$MLPERF_ROOT/data/C4_merged"
PREPROCESSED_PATH="$MLPERF_ROOT/data/C4_processed"
TOKENIZER_PATH="$MLPERF_ROOT/models/Llama-3.1-8B"

mkdir -p "$PREPROCESSED_PATH"

test -f "$CONTAINER" || {
    echo "ERROR: container not found: $CONTAINER"
    exit 1
}

test -d "$MERGED_C4_PATH" || {
    echo "ERROR: merged C4 not found: $MERGED_C4_PATH"
    exit 1
}

test -d "$TOKENIZER_PATH" || {
    echo "ERROR: tokenizer not found: $TOKENIZER_PATH"
    exit 1
}

# ----------------------------------------------------------------------
# CPU-aware preprocessing configuration
#
# WORKERS:
#   Number of workers passed to NeMo preprocess_data_for_megatron.py
#
# MAX_PARALLEL_JOBS:
#   Maximum number of preprocessing jobs allowed to run simultaneously
#
# Example for a 64-CPU allocation:
#
#   WORKERS=16 MAX_PARALLEL_JOBS=4 \
#     ./scripts/05_preprocess_c4_singularity.sh
#
# This keeps the approximate total worker count near 64.
# ----------------------------------------------------------------------

NPROC="$(nproc)"

MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"

if [ -z "${WORKERS:-}" ]; then
    WORKERS=$((NPROC / MAX_PARALLEL_JOBS))
    if [ "$WORKERS" -lt 1 ]; then
        WORKERS=1
    fi
fi

if [ "$WORKERS" -lt 1 ]; then
    echo "ERROR: WORKERS must be >= 1"
    exit 1
fi

if [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
    echo "ERROR: MAX_PARALLEL_JOBS must be >= 1"
    exit 1
fi

TOTAL_REQUESTED_WORKERS=$((WORKERS * MAX_PARALLEL_JOBS))

echo "===== C4 PREPROCESS CONFIGURATION ====="
echo "MLPERF_ROOT       : $MLPERF_ROOT"
echo "CONTAINER         : $CONTAINER"
echo "MERGED_C4_PATH    : $MERGED_C4_PATH"
echo "PREPROCESSED_PATH : $PREPROCESSED_PATH"
echo "TOKENIZER_PATH    : $TOKENIZER_PATH"
echo "Detected CPUs     : $NPROC"
echo "WORKERS/job       : $WORKERS"
echo "Parallel jobs     : $MAX_PARALLEL_JOBS"
echo "Worker upper bound: $TOTAL_REQUESTED_WORKERS"

if [ "$TOTAL_REQUESTED_WORKERS" -gt "$NPROC" ]; then
    echo
    echo "WARNING:"
    echo "WORKERS * MAX_PARALLEL_JOBS exceeds detected CPU count."
    echo "Performance may degrade due to oversubscription."
fi

echo

run_preprocess()
{
    local input_file="$1"
    local output_prefix="$2"
    local log_file="$3"

    singularity exec \
      -B "$MERGED_C4_PATH:/dataset" \
      -B "$PREPROCESSED_PATH:/outputs" \
      -B "$TOKENIZER_PATH:/tokenizer" \
      "$CONTAINER" \
      /bin/bash -lc "
        python3 /workspace/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
          --input /dataset/${input_file} \
          --output-prefix /outputs/${output_prefix} \
          --tokenizer-library huggingface \
          --tokenizer-type /tokenizer \
          --dataset-impl mmap \
          --workers ${WORKERS}
      " \
      > "$log_file" 2>&1
}

wait_for_slot()
{
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]; do
        sleep 2
    done
}

echo "===== PREPROCESS TRAIN SHARDS ====="

for index in {0..7}; do

    input_file="c4-train.en_${index}.json.gz"
    output_prefix="c4-train.en_${index}"
    log_file="$PREPROCESSED_PATH/preprocess_en_${index}.log"

    test -f "$MERGED_C4_PATH/$input_file" || {
        echo "ERROR: missing input: $MERGED_C4_PATH/$input_file"
        exit 1
    }

    wait_for_slot

    echo "Starting en_${index}"

    run_preprocess \
      "$input_file" \
      "$output_prefix" \
      "$log_file" &

done

echo
echo "===== PREPROCESS VALIDATION ====="

validation_input="c4-validation-91205-samples.en.json.gz"
validation_output="c4-validation-91205-samples.en"
validation_log="$PREPROCESSED_PATH/preprocess_validation.log"

test -f "$MERGED_C4_PATH/$validation_input" || {
    echo "ERROR: missing validation input: $MERGED_C4_PATH/$validation_input"
    exit 1
}

wait_for_slot

echo "Starting validation"

run_preprocess \
  "$validation_input" \
  "$validation_output" \
  "$validation_log" &

echo
echo "===== WAIT FOR ALL PREPROCESS JOBS ====="

wait

echo
echo "===== VERIFY PREPROCESSED DATA ====="

find "$PREPROCESSED_PATH" \
  -maxdepth 1 \
  \( -name '*.bin' -o -name '*.idx' \) \
  -printf '%f %s bytes\n' \
  | sort

echo
echo "===== SIZE ====="
du -sh "$PREPROCESSED_PATH"

echo
echo "===== SUCCESS ====="
