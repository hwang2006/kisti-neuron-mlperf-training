#!/bin/bash
set -e

REAL_USER="${LOGNAME:-$(id -un)}"
export MLPERF_ROOT="${MLPERF_ROOT:-/scratch/${REAL_USER}/mlperf}"

mkdir -p \
    "$MLPERF_ROOT/containers" \
    "$MLPERF_ROOT/data/C4" \
    "$MLPERF_ROOT/data/C4_merged" \
    "$MLPERF_ROOT/data/C4_processed" \
    "$MLPERF_ROOT/models/Llama-3.1-8B" \
    "$MLPERF_ROOT/checkpoints/continual" \
    "$MLPERF_ROOT/logs"

echo "===== MLPerf directory layout ====="
echo "MLPERF_ROOT=$MLPERF_ROOT"
echo

find "$MLPERF_ROOT" \
    -maxdepth 2 \
    -type d \
    | sort
