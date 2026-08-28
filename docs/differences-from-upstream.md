# Differences from the Official MLCommons Training Repository

This document describes how the KISTI NEURON reproduction workflow differs
from the official MLCommons Training reference implementation for the
Small LLM / Llama 3.1 8B workload.

The goal of the KISTI repository is not to replace or redefine the MLPerf
workload. It provides the environment adaptation, configuration, launch
automation, preliminary validation support, and compatibility fixes required
to reproduce and study the workload on the KISTI NEURON H200 system.

---

## 1. Upstream Reference

Official repository:

    https://github.com/mlcommons/training

Upstream workload:

    small_llm_pretraining/nemo

Pinned upstream commit:

    aa344c7fb900e82ed19fb94aebfed50c63ab2204

The official MLCommons source is cloned under:

    $MLPERF_ROOT/training

The Small LLM implementation is therefore located at:

    $MLPERF_ROOT/training/small_llm_pretraining/nemo

Important upstream files include:

    Dockerfile.h200
    Dockerfile.b200
    Dockerfile.mi325

    config_H100_1x8x4_8b.sh
    config_H200_1x8x1_8b.sh
    config_MI325X_1x8x1_8b.sh

    pretrain_llama31.py
    run_llama31.sh

    callbacks.py
    requirements.txt
    utils/
    patches/

The KISTI preparation workflow additionally installs:

    pretrain_llama31_kisti.py
    run_llama31_kisti.sh

These KISTI variants are not part of the original upstream checkout.

---

## 2. Overall Relationship

The relationship between the two repositories is:

    Official MLCommons Training
        |
        | defines the workload and reference implementation
        |
        v
    KISTI NEURON reproduction repository
        |
        | adapts the execution environment
        | adds reproducibility automation
        | adds two-H200 experiment configurations
        | adds compatibility fixes
        | adds preliminary validation support
        |
        v
    KISTI NEURON H200 execution

The KISTI repository therefore acts as a system-integration and
reproducibility layer around the official workload.

---

## 3. Container Runtime: Docker to Singularity

### Official MLCommons workflow

The official Small LLM implementation provides Dockerfiles such as:

    Dockerfile.h200

The reference workflow assumes a Docker-compatible environment for building
and running the benchmark container.

Conceptually:

    Dockerfile.h200
        |
        +-- Docker image
        |
        +-- run_llama31.sh
        |
        +-- pretrain_llama31.py
        |
        +-- NeMo / Megatron Core

### KISTI NEURON workflow

Docker is not available on the KISTI NEURON compute nodes.

Therefore the KISTI workflow converts the container build and execution
procedure to Singularity.

The KISTI repository provides:

    containers/mlperf-h200.def

which is used to build:

    $MLPERF_ROOT/containers/mlperf-llama31-h200.sif

using:

    singularity build --fakeroot \
        "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif" \
        containers/mlperf-h200.def

The benchmark is then executed with:

    singularity exec --nv ...

rather than Docker.

Singularity's `--nv` option exposes the NVIDIA devices and host driver
libraries required by the container.

This is an execution-environment adaptation. It does not modify the
Llama 3.1 8B model or the MLPerf convergence target.

---

## 4. Official and KISTI Execution Paths

### Official reference path

The upstream path is approximately:

    Dockerfile.h200
        |
        +-- Docker container
        |
        +-- config_H200_1x8x1_8b.sh
        |
        +-- run_llama31.sh
        |
        +-- pretrain_llama31.py
        |
        +-- NeMo
        |
        +-- Megatron Core

### KISTI NEURON path

The KISTI execution path is:

    containers/mlperf-h200.def
        |
        +-- Singularity SIF
        |
        +-- scripts/run_h200_tp1.sh
        |       or
        +-- scripts/run_h200_tp2.sh
        |
        +-- scripts/run_h200.sh
        |
        +-- configs/h200/h200_2gpu_tp1_dp2.sh
        |       or
        +-- configs/h200/h200_2gpu_tp2_dp1.sh
        |
        +-- run_llama31_kisti.sh
        |
        +-- pretrain_llama31_kisti.py
        |
        +-- NeMo
        |
        +-- Megatron Core

The KISTI scripts wrap and adapt the upstream workload rather than
reimplementing the training stack.

---

## 5. Two-H200 Experiment Configuration

The upstream repository includes:

    config_H200_1x8x1_8b.sh

The current KISTI validation experiments use two H200 GPUs on a single
NEURON node.

Two configurations are provided:

    configs/h200/h200_2gpu_tp1_dp2.sh
    configs/h200/h200_2gpu_tp2_dp1.sh

### TP1 / DP2

    GPUs = 2
    TP   = 1
    DP   = 2
    MBS  = 1
    GBS  = 32

### TP2 / DP1

    GPUs = 2
    TP   = 2
    DP   = 1
    MBS  = 1
    GBS  = 32

These configurations allow controlled comparison of tensor-parallel and
data-parallel execution on the same two-H200 allocation.

---

## 6. KISTI Runtime Variants

### Upstream

    pretrain_llama31.py
    run_llama31.sh

### KISTI

    patches/pretrain_llama31_kisti.py
    patches/run_llama31_kisti.sh

During preparation, the KISTI variants are copied into:

    $MLPERF_ROOT/training/small_llm_pretraining/nemo

as:

    pretrain_llama31_kisti.py
    run_llama31_kisti.sh

The original upstream files remain available so that the changes can be
inspected directly.

---

## 7. Preliminary Validation with the Last 256 C4 Shards

The KISTI implementation adds:

    --use_last_256_shards

This option was introduced as a preliminary validation mode before performing
a full-dataset benchmark run.

The purpose was to verify that the complete end-to-end workflow operated
correctly on KISTI NEURON before committing to a full C4 benchmark execution.

The preliminary validation checks include:

- container construction and execution
- Singularity GPU access
- C4 dataset access
- tokenizer loading
- NeMo initialization
- Megatron Core initialization
- distributed multi-GPU training
- MLPerf logging
- convergence behavior
- long-running job stability
- result and log generation

The original C4 training data contains 1024 raw shards.

In the consolidated KISTI preprocessing layout:

    c4-train.en_0  <- raw shards   0-127
    c4-train.en_1  <- raw shards 128-255
    c4-train.en_2  <- raw shards 256-383
    c4-train.en_3  <- raw shards 384-511
    c4-train.en_4  <- raw shards 512-639
    c4-train.en_5  <- raw shards 640-767
    c4-train.en_6  <- raw shards 768-895
    c4-train.en_7  <- raw shards 896-1023

Therefore the final 256 raw shards correspond to:

    c4-train.en_6
    c4-train.en_7

The KISTI code explicitly selects these two consolidated datasets when
`--use_last_256_shards` is enabled.

This reduced-data mode is a preliminary validation mechanism.

It was not introduced to redefine the official MLPerf dataset or to replace
the intended full-dataset benchmark configuration.

After validating the end-to-end workflow, the intended next step is to run
the workload using the full dataset configuration.

---

## 8. Source-Level Changes in `pretrain_llama31_kisti.py`

The KISTI variant adds support for:

    --use_last_256_shards

and explicitly maps that option to:

    c4-train.en_6_text_document
    c4-train.en_7_text_document

Additional changes include limited checkpoint-handling adjustments used by
the KISTI experiment workflow.

The exact source-level differences are recorded in:

    reproducibility/pretrain_llama31_kisti.diff

---

## 9. Source-Level Changes in `run_llama31_kisti.sh`

The KISTI variant adds:

- `INITIAL_CKPT` handling
- explicit propagation of `MAX_LR`
- `--use_last_256_shards`
- invocation of `pretrain_llama31_kisti.py`

The exact source-level differences are recorded in:

    reproducibility/run_llama31_kisti.diff

---

## 10. Python `pangu` Compatibility Fix

The KISTI repository contains:

    patches/pythonfix/pangu.py

This file is not part of the official MLCommons Small LLM reference
implementation.

During execution of the selected NVIDIA PyTorch / NeMo environment, NeMo
imports:

    from pangu import spacing

The `pangu` package installed in the container does not expose the API
expected by this version of NeMo.

Without the compatibility fix, startup fails with:

    ImportError: cannot import name 'spacing' from 'pangu'

The KISTI launcher therefore prepends:

    $REPO_ROOT/patches/pythonfix

to:

    PYTHONPATH

so that the compatible:

    patches/pythonfix/pangu.py

is loaded before the incompatible container-installed package.

This is an environment compatibility workaround.

It does not modify:

- the Llama model architecture
- the optimizer
- the training algorithm
- the C4 data itself
- the global batch size
- the MLPerf convergence criterion

---

## 11. Dataset File Layout

The Megatron binary dataset files used by this workflow include:

    c4-train.en_6_text_document.bin
    c4-train.en_6_text_document.idx
    c4-train.en_7_text_document.bin
    c4-train.en_7_text_document.idx

Validation uses:

    c4-validation-91205-samples.en_text_document.bin
    c4-validation-91205-samples.en_text_document.idx

Because the complete preprocessed C4 dataset requires several hundred GB,
the KISTI workflow supports either:

    downloading / preprocessing the data

or:

    reusing an existing validated dataset through symbolic links

The symbolic-link mechanism is a storage optimization and does not alter the
underlying benchmark input.

---

## 12. Long-Running Job Handling

Small LLM training on two H200 GPUs can require many hours.

The KISTI launcher uses:

    nohup

to keep the benchmark process running after an interactive SSH session is
disconnected.

For example:

    ./scripts/run_h200_tp1.sh

starts the Singularity execution in the background and reports:

    PID=<pid>
    LOG=<launcher-log>

The underlying Slurm allocation must remain active for the entire run.

`nohup` protects against loss of the interactive shell but does not extend
the Slurm allocation time.

---

## 13. Reproducibility Automation

The KISTI repository provides preparation and launch scripts that are not part
of the upstream MLCommons repository.

Representative files include:

    scripts/01_prepare_directories.sh
    scripts/02_prepare_mlcommons_source.sh
    scripts/03_download_preprocessed_data.sh

    scripts/run_h200.sh
    scripts/run_h200_tp1.sh
    scripts/run_h200_tp2.sh

    configs/h200/h200_2gpu_tp1_dp2.sh
    configs/h200/h200_2gpu_tp2_dp1.sh

    containers/mlperf-h200.def

    patches/pretrain_llama31_kisti.py
    patches/run_llama31_kisti.sh
    patches/pythonfix/pangu.py

The objective is to allow a new NEURON user to reproduce the experiment from
a clean working directory without depending on undocumented shell history or
previous local state.

---

## 14. Source-Level Reproducibility Records

Exact source differences from the pinned MLCommons reference are stored in:

- [`../reproducibility/pretrain_llama31_kisti.diff`](../reproducibility/pretrain_llama31_kisti.diff)
- [`../reproducibility/run_llama31_kisti.diff`](../reproducibility/run_llama31_kisti.diff)

These files allow the KISTI modifications to be inspected and audited
independently of this documentation.

---

## 15. What Remains Based on the Official Reference

The following major components remain based on the official MLCommons
reference workload:

- Llama 3.1 8B model
- NVIDIA NeMo training stack
- Megatron Core
- MLPerf logging callbacks
- C4 dataset
- tokenizer
- validation procedure
- convergence target
- general Small LLM benchmark structure

The KISTI repository primarily changes the surrounding execution,
configuration, compatibility, and reproducibility environment.

---

## 16. Preliminary Validation vs Full Benchmark

It is important to distinguish the current reduced-data validation workflow
from the intended full benchmark.

### Preliminary KISTI validation

    --use_last_256_shards

uses:

    en_6 + en_7

and is intended to validate the entire execution pipeline.

### Full benchmark

The next phase is to use the full dataset configuration after the execution
environment and workflow have been validated successfully.

Results produced by the preliminary validation configuration should therefore
be identified explicitly as validation/performance-engineering results rather
than being presented as an official MLCommons submission.

---

## 17. Official Submission Status

This repository is intended for:

- reproducibility testing
- environment validation
- performance engineering
- preparation for larger-scale MLPerf experiments

A successful run with this repository should not automatically be interpreted
as an official MLCommons submission.

Official submission requires compliance with the applicable MLPerf Training
rules, system description requirements, compliance tests, result packaging,
and MLCommons review process.

---

## 18. Summary

The main adaptations can be grouped into two categories.

### Environment adaptations

- Docker to Singularity
- KISTI NEURON storage and directory layout
- two-H200 configuration
- Singularity launch wrappers
- `nohup` background execution
- Python `pangu` compatibility fix
- reproducibility automation

### Preliminary validation support

- `--use_last_256_shards`
- explicit use of consolidated `en_6` and `en_7`
- intended for end-to-end validation before a full-dataset benchmark run

The guiding principle is to keep the official workload as intact as possible
while making every KISTI-specific change explicit, traceable, and
reproducible.
