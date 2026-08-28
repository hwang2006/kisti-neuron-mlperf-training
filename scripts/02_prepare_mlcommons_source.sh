#!/bin/bash
set -e

REAL_USER="${LOGNAME:-$(id -un)}"
export MLPERF_ROOT="${MLPERF_ROOT:-/scratch/${REAL_USER}/mlperf}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MLCOMMONS_REPO="https://github.com/mlcommons/training.git"
MLCOMMONS_COMMIT="aa344c7fb900e82ed19fb94aebfed50c63ab2204"

TRAIN_ROOT="$MLPERF_ROOT/training"
NEMO_ROOT="$TRAIN_ROOT/small_llm_pretraining/nemo"

echo "===== MLCommons Training Source Setup ====="
echo "MLPERF_ROOT : $MLPERF_ROOT"
echo "Repository  : $MLCOMMONS_REPO"
echo "Commit      : $MLCOMMONS_COMMIT"
echo

if [ ! -d "$TRAIN_ROOT/.git" ]; then
    echo "Cloning MLCommons Training..."
    git clone "$MLCOMMONS_REPO" "$TRAIN_ROOT"
else
    echo "Existing MLCommons repository found:"
    echo "$TRAIN_ROOT"
fi

cd "$TRAIN_ROOT"

echo
echo "Fetching required commit..."
git fetch origin

echo
echo "Checking out reproducibility commit..."
git checkout "$MLCOMMONS_COMMIT"

echo
echo "HEAD:"
git rev-parse HEAD

CURRENT_COMMIT=$(git rev-parse HEAD)

if [ "$CURRENT_COMMIT" != "$MLCOMMONS_COMMIT" ]; then
    echo "ERROR: checkout mismatch"
    echo "Expected: $MLCOMMONS_COMMIT"
    echo "Actual  : $CURRENT_COMMIT"
    exit 1
fi

echo
echo "===== INSTALL KISTI NEURON PATCHES ====="

cp -f \
    "$REPO_ROOT/patches/pretrain_llama31_kisti.py" \
    "$NEMO_ROOT/pretrain_llama31_kisti.py"

cp -f \
    "$REPO_ROOT/patches/run_llama31_kisti.sh" \
    "$NEMO_ROOT/run_llama31_kisti.sh"

chmod +x "$NEMO_ROOT/run_llama31_kisti.sh"

echo
echo "===== VERIFY ====="

test -f "$NEMO_ROOT/pretrain_llama31_kisti.py"
test -x "$NEMO_ROOT/run_llama31_kisti.sh"

/bin/bash -n "$NEMO_ROOT/run_llama31_kisti.sh"

python3 -m py_compile \
    "$NEMO_ROOT/pretrain_llama31_kisti.py"

echo
echo "Installed:"
ls -lh \
    "$NEMO_ROOT/pretrain_llama31_kisti.py" \
    "$NEMO_ROOT/run_llama31_kisti.sh"

echo
echo "===== SUCCESS ====="
echo "MLCommons source and KISTI patches are ready."

# ----------------------------------------------------------------------
# KISTI A100 compatibility sources
# ----------------------------------------------------------------------

A100_PRETRAIN="$REPO_ROOT/patches/pretrain_llama31_a100.py"
A100_RUNNER="$REPO_ROOT/patches/run_llama31_a100.sh"

if [ -f "$A100_PRETRAIN" ]; then
    cp -f "$A100_PRETRAIN" \
        "$NEMO_DIR/pretrain_llama31_a100.py"
fi

if [ -f "$A100_RUNNER" ]; then
    cp -f "$A100_RUNNER" \
        "$NEMO_DIR/run_llama31_a100.sh"
    chmod +x "$NEMO_DIR/run_llama31_a100.sh"
fi
