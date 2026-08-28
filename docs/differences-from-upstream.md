# Differences from the Official MLCommons Training Repository

This document describes how the KISTI NEURON reproduction workflow differs
from the official MLCommons Training reference implementation for the
Small LLM / Llama 3.1 8B workload.

The goal of this repository is not to replace or redefine the MLPerf workload.
It provides environment adaptation, hardware-specific configuration, launch
automation, compatibility fixes, preliminary validation support, and
reproducibility metadata for NVIDIA GPU systems on KISTI NEURON.

The currently documented reproduced configurations are:

- NVIDIA H200: 2 GPUs, TP1/DP2
- NVIDIA H200: 2 GPUs, TP2/DP1
- NVIDIA A100: 8 GPUs, TP4/DP2

---

## 1. Upstream Reference

Official repository:

```text
https://github.com/mlcommons/training
```

Upstream workload:

```text
small_llm_pretraining/nemo
```

Pinned upstream commit:

```text
aa344c7fb900e82ed19fb94aebfed50c63ab2204
```

The official source is cloned under:

```text
$MLPERF_ROOT/training
```

The Small LLM implementation is located at:

```text
$MLPERF_ROOT/training/small_llm_pretraining/nemo
```

The KISTI preparation workflow additionally installs hardware-specific runtime
variants into the upstream workload directory.

H200 path:

```text
pretrain_llama31_kisti.py
run_llama31_kisti.sh
```

A100 path:

```text
pretrain_llama31_a100.py
run_llama31_a100.sh
```

---

## 2. Overall Relationship

```text
Official MLCommons Training
        |
        | defines workload and reference implementation
        v
KISTI NEURON reproduction repository
        |
        |-- adapts Docker-oriented execution to Singularity
        |-- adds H200 and A100 experiment configurations
        |-- adds KISTI launch automation
        |-- adds runtime compatibility fixes
        |-- adds preliminary validation support
        |-- records reproducibility metadata
        v
KISTI NEURON NVIDIA GPU execution
```

The KISTI repository acts as a system-integration and reproducibility layer
around the official workload.

---

## 3. Container Runtime: Docker to Singularity

KISTI NEURON compute nodes use Singularity rather than Docker.

The repository provides:

```text
containers/mlperf-h200.def
```

which is used to build:

```text
$MLPERF_ROOT/containers/mlperf-llama31-h200.sif
```

The current image name contains `h200` for historical reasons. The same
container environment is also used by the documented A100 reproduction path.

This is an execution-environment adaptation and does not redefine the model,
training algorithm, dataset, or convergence target.

---

## 4. KISTI Execution Paths

### H200

```text
scripts/run_h200_tp1.sh or scripts/run_h200_tp2.sh
        |
        +-- scripts/run_h200.sh
                |
                +-- configs/h200/h200_2gpu_tp1_dp2.sh
                |   or
                +-- configs/h200/h200_2gpu_tp2_dp1.sh
                        |
                        +-- run_llama31_kisti.sh
                                |
                                +-- pretrain_llama31_kisti.py
```

### A100

```text
scripts/run_a100_tp4_dp2.sh
        |
        +-- scripts/run_a100.sh
                |
                +-- configs/a100/a100_8gpu_tp4_dp2.sh
                        |
                        +-- run_llama31_a100.sh
                                |
                                +-- pretrain_llama31_a100.py
```

The KISTI scripts wrap and adapt the upstream workload rather than
reimplementing the training stack.

---

## 5. Hardware-Specific Configurations

### H200: 2 GPUs, TP1 / DP2

```text
GPUs = 2
TP   = 1
DP   = 2
MBS  = 1
GBS  = 32
```

### H200: 2 GPUs, TP2 / DP1

```text
GPUs = 2
TP   = 2
DP   = 1
MBS  = 1
GBS  = 32
```

### A100: 8 GPUs, TP4 / DP2

```text
GPUs = 8
TP   = 4
DP   = 2
MBS  = 1
GBS  = 32
```

Important A100-specific settings are:

```bash
DISABLE_FP8=1
ENABLE_RECOMPUTE=0
DISABLE_CE_FUSION=0
```

---

## 6. KISTI Runtime Variants

H200 variants:

```text
patches/pretrain_llama31_kisti.py
patches/run_llama31_kisti.sh
```

A100 variants:

```text
patches/pretrain_llama31_a100.py
patches/run_llama31_a100.sh
```

During preparation these files are copied into:

```text
$MLPERF_ROOT/training/small_llm_pretraining/nemo
```

---

## 7. Shared Dataset and Tokenizer

The H200 and A100 reproductions reuse the same validated preprocessed C4
dataset and the same Llama 3.1 tokenizer.

Typical shared paths are:

```text
$MLPERF_ROOT/data/C4_processed
$MLPERF_ROOT/models/Llama-3.1-8B
```

Symbolic links may be used to reuse an existing validated dataset and tokenizer.

---

## 8. Experiment-Specific NPY / Dataset Index Cache

Megatron/NeMo may generate runtime document, sample, and shuffle indices.

Separate cache directories are used per experiment, for example:

```text
$MLPERF_ROOT/data/npy_indices/h200_2gpu_tp1_dp2
$MLPERF_ROOT/data/npy_indices/h200_2gpu_tp2_dp1
$MLPERF_ROOT/data/npy_indices/a100_8gpu_tp4_dp2
```

The separation is primarily for reproducibility and experiment isolation, not
simply because the GPU generation differs.

---

## 9. Preliminary Validation with the Last 256 C4 Shards

The KISTI runtime supports:

```text
--use_last_256_shards
```

This is used for preliminary validation before a full-dataset benchmark run.

In the consolidated KISTI preprocessing layout, the final 256 raw shards map to:

```text
c4-train.en_6
c4-train.en_7
```

This reduced-data mode validates container execution, GPU access, dataset and
tokenizer loading, NeMo/Megatron initialization, distributed training, MLPerf
logging, convergence behavior, long-running stability, and result generation.

It does not redefine the official MLPerf Training dataset.

Results produced with this mode should be described as preliminary,
reproduction, or performance-engineering results rather than official
full-dataset MLPerf submissions.

---

## 10. H200 Source-Level Changes

`pretrain_llama31_kisti.py` adds support for:

```text
--use_last_256_shards
```

and maps it to:

```text
c4-train.en_6_text_document
c4-train.en_7_text_document
```

`run_llama31_kisti.sh` adds or propagates:

- `INITIAL_CKPT`
- `MAX_LR`
- `--use_last_256_shards`
- invocation of `pretrain_llama31_kisti.py`

Exact H200 source differences are recorded in:

```text
reproducibility/pretrain_llama31_kisti.diff
reproducibility/run_llama31_kisti.diff
```

---

## 11. A100-Specific Runtime Controls

The final documented A100 TP4/DP2 reproduction uses:

```bash
DISABLE_FP8=1
ENABLE_RECOMPUTE=0
DISABLE_CE_FUSION=0
```

The A100 code retains optional controls for activation recomputation and
cross-entropy fusion for debugging and experimentation.

These optional controls are not enabled in the final documented TP4/DP2
configuration.

See `docs/a100-notes.md`.

---

## 12. Python `pangu` Compatibility Fix

The repository contains:

```text
patches/pythonfix/pangu.py
```

The KISTI launcher prepends this directory to `PYTHONPATH` to avoid:

```text
ImportError: cannot import name 'spacing' from 'pangu'
```

This is an environment compatibility workaround and does not modify the model,
optimizer, training algorithm, dataset, global batch size, or convergence
criterion.

---

## 13. Long-Running Job Handling

The KISTI launchers use `nohup` so that the benchmark process can continue
after an interactive SSH shell disconnects.

The underlying Slurm allocation must remain active for the entire run.

---

## 14. Reproducibility Automation

Representative files include:

```text
scripts/01_prepare_directories.sh
scripts/02_prepare_mlcommons_source.sh
scripts/03_download_preprocessed_data.sh

scripts/run_h200.sh
scripts/run_h200_tp1.sh
scripts/run_h200_tp2.sh

scripts/run_a100.sh
scripts/run_a100_tp4_dp2.sh

configs/h200/h200_2gpu_tp1_dp2.sh
configs/h200/h200_2gpu_tp2_dp1.sh
configs/a100/a100_8gpu_tp4_dp2.sh
```

The objective is to allow a new NEURON user to reproduce the experiment from
a clean working directory without depending on undocumented shell history.

---

## 15. What Remains Based on the Official Reference

The following remain based on the official MLCommons reference workload:

- Llama 3.1 8B model
- NVIDIA NeMo training stack
- Megatron Core
- MLPerf logging callbacks
- C4 dataset
- tokenizer
- validation procedure
- convergence target
- general Small LLM workload structure

---

## 16. Preliminary Validation vs Full Benchmark

The current preliminary reproduction path uses:

```text
--use_last_256_shards
```

A full-dataset run should disable the preliminary last-256-shard selection and
be validated separately.

---

## 17. Official Submission Status

This repository is intended for reproducibility testing, environment
validation, performance engineering, and preparation for larger-scale MLPerf
experiments.

A successful run should not automatically be interpreted as an official
MLCommons submission.

Official submission requires compliance with the applicable MLPerf Training
rules, system-description requirements, compliance tests, result packaging,
and MLCommons review process.

---

## 18. Summary

### Environment adaptations

- Docker to Singularity
- KISTI NEURON directory and storage layout
- Singularity launch wrappers
- `nohup` background execution
- Python `pangu` compatibility fix
- reproducibility automation

### Hardware-specific configurations

- H200 2-GPU TP1/DP2
- H200 2-GPU TP2/DP1
- A100 8-GPU TP4/DP2
- A100 FP8 control and related runtime options

### Preliminary validation support

- `--use_last_256_shards`
- explicit use of consolidated `en_6` and `en_7`
- intended for end-to-end validation before a full-dataset benchmark run

The guiding principle is to keep the official workload as intact as possible
while making every KISTI-specific change explicit, traceable, and reproducible.
