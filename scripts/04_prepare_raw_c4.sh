#!/bin/bash
set -e

REAL_USER="${LOGNAME:-$(id -un)}"
export MLPERF_ROOT="${MLPERF_ROOT:-/scratch/${REAL_USER}/mlperf}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export C4_PATH="$MLPERF_ROOT/data/C4"
export MERGED_C4_PATH="$MLPERF_ROOT/data/C4_merged"
export N_VALIDATION_SAMPLES=91205

mkdir -p "$C4_PATH"
mkdir -p "$MERGED_C4_PATH"

echo "===== DOWNLOAD RAW C4 ====="

bash <(curl -s \
  https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "$C4_PATH" \
  https://training.mlcommons-storage.org/metadata/c4-full-dataset-unzipped.uri

echo
echo "===== COMPRESS JSON -> JSON.GZ ====="

export C4_PATH
/bin/bash \
  "$REPO_ROOT/scripts/upstream-data-utils/parallel_compress_json_to_gz.sh"

echo
echo "===== CONSOLIDATE 1024 TRAIN SHARDS -> 8 GROUPS ====="

export C4_PATH
export MERGED_C4_PATH
export N_VALIDATION_SAMPLES

/bin/bash \
  "$REPO_ROOT/scripts/upstream-data-utils/consolidate_data.sh"

echo
echo "===== VERIFY MERGED DATA ====="

ls -lh "$MERGED_C4_PATH"

echo
echo "===== SIZE ====="
du -sh "$C4_PATH" "$MERGED_C4_PATH"
