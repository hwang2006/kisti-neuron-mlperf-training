#!/bin/bash

# KISTI NEURON
# MLPerf Training Small LLM / Llama 3.1 8B
# NVIDIA A100 x8
#
# Historical reproduced configuration:
#   Tensor Parallelism = 4
#   Data Parallelism   = 2
#
# This configuration reproduces the earlier KISTI A100 run.
# The historical A100 runner used --use_last_256_shards.

: "${MLPERF_ROOT:?MLPERF_ROOT must be set}"

# ----------------------------------------------------------------------
# Local execution placeholders required by the MLCommons-derived runner
# ----------------------------------------------------------------------

export USER="${USER:-$(id -un)}"
export HOST="${HOST:-localhost}"
export ACCOUNT="${ACCOUNT:-DUMMY}"
export PARTITION="${PARTITION:-DUMMY}"
export REMOTE=0

# ----------------------------------------------------------------------
# System configuration
# ----------------------------------------------------------------------

export NNODES=1
export GPUS_PER_NODE=8

export SIZE="8b"

# Parallelism
export TENSOR_PARALLEL_SIZE=4
export PIPELINE_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1

# TP4 x DP2 = 8 GPUs

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

export GBS=32
export MBS=1

export MAX_STEPS=1200000
export WARMUP_STEPS=512
export MAX_LR="5e-4"

export EVAL_EVERY=12288
export START_EVAL_AT=0

export NEXP=1
export NPAR=1
export START_STEPS=0

# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------

export SAVE_CKPT=0
export USE_CKPT=0
export INITIAL_CKPT=""
export FROM_HF=0

export CONTINUAL_CKPT="$MLPERF_ROOT/checkpoints/a100_8gpu_tp4_dp2"

# ----------------------------------------------------------------------
# A100-specific settings recovered from the historical successful run
# ----------------------------------------------------------------------

# A100 does not use the H200 FP8 configuration used by the reference path.
export DISABLE_FP8=1

export ENABLE_RECOMPUTE=0

# Historical final TP4/DP2 run used CE fusion enabled.
export DISABLE_CE_FUSION=0

# ----------------------------------------------------------------------
# Shared dataset/tokenizer
# ----------------------------------------------------------------------

export TOKENIZER_PATH="$MLPERF_ROOT/models/Llama-3.1-8B"
export PREPROCESSED_PATH="$MLPERF_ROOT/data/C4_processed"

# Per-experiment dataset index cache.
export TMP_NPY_INDEX="$MLPERF_ROOT/data/npy_indices/a100_8gpu_tp4_dp2"

# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

export JOB_DIR="$MLPERF_ROOT/logs/mlperf-a100-8gpu-tp4-dp2"

export TRITON_LIBCUDA_PATH="/.singularity.d/libs"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export DGXSYSTEM="KISTI_NEURON_A100_8GPU_TP4_DP2"
