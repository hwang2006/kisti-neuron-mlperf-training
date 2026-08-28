# MLPerf Llama 3.1 8B Singularity Container

This directory preserves the Singularity build context used for the
KISTI NEURON H200 MLPerf Training experiments.

## Base image

The container is bootstrapped from:

    nvcr.io/nvidia/pytorch:25.01-py3

## Pinned software

The definition file installs:

- NVIDIA NeMo v2.1.0
- NeMo-Run v0.4.0
- Megatron-Core core_r0.11.0

The reference container reported:

- Python 3.12.3
- PyTorch 2.6.0a0+ecf3bae40a.nv25.01
- CUDA 12.8
- NeMo 2.1.0
- Megatron-Core 0.11.0rc0
- Transformer Engine 1.14.0+87fbe81
- NCCL 2.25.1

## Historical build command

The successful KISTI NEURON image was built with:

    cd containers

    singularity build --fakeroot \
        /scratch/$USER/mlperf/containers/mlperf-llama31-h200.sif \
        mlperf-h200.def

For this repository, use:

    export MLPERF_ROOT=/scratch/$USER/mlperf
    ./scripts/06_build_container.sh

## Build context

The Singularity definition copies the following into `/workspace/code`:

- requirements.txt
- callbacks.py
- pretrain_llama31.py
- run_llama31.sh
- config_H200_1x8x1_8b.sh
- config_H200_1x2x1_8b.sh
- utils/
- patches/

These files are preserved from the MLCommons Training source tree used
for the experiment.

## KISTI runtime wrappers

The successful H200 runs used host-side KISTI-specific files:

- patches/pretrain_llama31_kisti.py
- patches/run_llama31_kisti.sh

They are installed into the checked-out MLCommons source by:

    scripts/02_prepare_mlcommons_source.sh

## Reference image hashes

The approximately 14 GB SIF image itself is not stored in Git.

Reference SHA256 values are recorded in:

    reproducibility/container_reference_sha256.txt

A newly rebuilt SIF is not necessarily expected to be byte-identical
because external package repositories and image metadata may change.
Software versions and benchmark behavior should therefore also be
verified.
