#!/bin/bash
set -e

REAL_USER="${LOGNAME:-$(id -un)}"
export MLPERF_ROOT="${MLPERF_ROOT:-/scratch/${REAL_USER}/mlperf}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BUILD_CONTEXT="$REPO_ROOT/containers"
DEF_FILE="$BUILD_CONTEXT/mlperf-h200.def"

CONTAINER_DIR="$MLPERF_ROOT/containers"
OUTPUT_SIF="$CONTAINER_DIR/mlperf-llama31-h200.sif"

mkdir -p "$CONTAINER_DIR"

echo "===== SINGULARITY ====="
singularity --version

echo
echo "===== BUILD CONTEXT ====="
echo "$BUILD_CONTEXT"

echo
echo "===== DEFINITION ====="
echo "$DEF_FILE"

echo
echo "===== OUTPUT ====="
echo "$OUTPUT_SIF"

echo
echo "===== CHECK REQUIRED BUILD FILES ====="

required_files=(
    requirements.txt
    callbacks.py
    pretrain_llama31.py
    run_llama31.sh
    config_H200_1x8x1_8b.sh
    config_H200_1x2x1_8b.sh
)

for f in "${required_files[@]}"; do
    if [ ! -f "$BUILD_CONTEXT/$f" ]; then
        echo "ERROR: missing build file: $BUILD_CONTEXT/$f"
        exit 1
    fi
    echo "OK: $f"
done

test -d "$BUILD_CONTEXT/utils" || {
    echo "ERROR: missing $BUILD_CONTEXT/utils"
    exit 1
}

test -d "$BUILD_CONTEXT/patches" || {
    echo "ERROR: missing $BUILD_CONTEXT/patches"
    exit 1
}

echo "OK: utils/"
echo "OK: patches/"

echo
echo "===== BUILD ====="
echo "Base image: nvcr.io/nvidia/pytorch:25.01-py3"
echo

# %files paths in the Singularity definition are resolved from
# the build context, so build from this directory.
cd "$BUILD_CONTEXT"

singularity build --fakeroot \
    "$OUTPUT_SIF" \
    "$DEF_FILE"

echo
echo "===== VERIFY IMAGE ====="

ls -lh "$OUTPUT_SIF"

echo
echo "SHA256:"
sha256sum "$OUTPUT_SIF"

echo
echo "===== IMAGE METADATA ====="

singularity inspect "$OUTPUT_SIF" | head -80

echo
echo "===== SOFTWARE CHECK ====="

singularity exec --nv \
    "$OUTPUT_SIF" \
    /bin/bash -lc '
        python3 --version

        python3 -c "
import torch
print(\"PyTorch:\", torch.__version__)
print(\"CUDA:\", torch.version.cuda)
"

        python3 -c "
import nemo
print(\"NeMo:\", nemo.__version__)
"

        python3 -c "
import megatron.core
print(\"Megatron-Core:\", getattr(megatron.core, \"__version__\", \"unknown\"))
"

        python3 -c "
import transformer_engine
print(\"Transformer Engine:\", getattr(transformer_engine, \"__version__\", \"unknown\"))
"
    '

echo
echo "===== SUCCESS ====="
echo "$OUTPUT_SIF"
