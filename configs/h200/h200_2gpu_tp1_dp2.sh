#!/bin/bash

: "${MLPERF_ROOT:?MLPERF_ROOT must be set}"

# NeMo-Run / LocalExecutor placeholders
export USER="DUMMY"
export HOST="DUMMY"
export ACCOUNT="DUMMY"
export PARTITION="DUMMY"
export TIME="08:00:00"

# System
export NNODES=1
export GPUS_PER_NODE=2
export MAX_RETRIES=1

# Benchmark
export SIZE="8b"
export GBS=32
export MBS=1
export TENSOR_PARALLEL_SIZE=1

export MAX_STEPS=1200000
export WARMUP_STEPS=512
export MAX_LR="5e-4"

export EVAL_EVERY=12288
export START_EVAL_AT=0

# Experiment management
export NPAR=1
export NEXP=1
export START_STEPS=0

# Checkpoints disabled for formal performance run
export SAVE_CKPT=0
export USE_CKPT=0
export INITIAL_CKPT=""
export FROM_HF=0

# Paths
export PREPROCESSED_PATH="$MLPERF_ROOT/data/C4_processed"
export MERGED_C4_PATH="$MLPERF_ROOT/data/C4_merged"
export TOKENIZER_PATH="$MLPERF_ROOT/models/Llama-3.1-8B"

export TMP_NPY_INDEX="$MLPERF_ROOT/data/npy_indices/h200_2gpu_tp1_dp2_full"
export JOB_DIR="$MLPERF_ROOT/logs/mlperf-h200-tp1"
export CONTINUAL_CKPT="$MLPERF_ROOT/checkpoints/continual"

# Required placeholder for NeMo-Run configuration
export IMAGE="DUMMY"

# Runtime
export TRITON_LIBCUDA_PATH="/.singularity.d/libs"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export DGXSYSTEM="KISTI_NEURON_H200_2GPU_TP1"

export USE_LAST_256_SHARDS=0

export USE_FULL_DATASET=1
