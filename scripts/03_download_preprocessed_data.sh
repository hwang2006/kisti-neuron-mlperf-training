#!/bin/bash
set -e

REAL_USER="${LOGNAME:-$(id -un)}"
export MLPERF_ROOT="${MLPERF_ROOT:-/scratch/${REAL_USER}/mlperf}"

DATA_ROOT="$MLPERF_ROOT/data"
MODEL_ROOT="$MLPERF_ROOT/models"

PREPROCESSED_PATH="$DATA_ROOT/C4_processed"
TOKENIZER_PATH="$MODEL_ROOT/Llama-3.1-8B"

mkdir -p "$PREPROCESSED_PATH"
mkdir -p "$TOKENIZER_PATH"

echo "===== Download MLCommons preprocessed C4 ====="

bash <(curl -s \
  https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "$PREPROCESSED_PATH" \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

echo
echo "===== Download MLCommons Llama 3.1 8B tokenizer ====="

bash <(curl -s \
  https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "$TOKENIZER_PATH" \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

echo
echo "===== VERIFY DATA ====="

ls -lh "$PREPROCESSED_PATH" | head -30

echo
echo "===== VERIFY TOKENIZER ====="

ls -lh "$TOKENIZER_PATH" | head -30

echo
echo "===== SIZE ====="

du -sh "$PREPROCESSED_PATH" "$TOKENIZER_PATH"

echo
echo "===== SUCCESS ====="
echo "PREPROCESSED_PATH=$PREPROCESSED_PATH"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
