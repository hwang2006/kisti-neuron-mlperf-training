# KISTI NEURON MLPerf Training Reproduction — Llama 3.1 8B on NVIDIA GPUs

This repository provides a reproducible workflow for running the
**MLPerf Training Small LLM / Llama 3.1 8B workload** on the
[KISTI NEURON GPU cluster](https://www.ksc.re.kr/eng/resources/neuron) using NVIDIA GPUs.

The current repository documents reproduced configurations on:

- NVIDIA H200
- NVIDIA A100

The workflow is based on the official MLCommons Training repository and adds
KISTI NEURON-specific configuration, Singularity-based execution, runtime
compatibility fixes, hardware-specific launch scripts, and reproducibility
metadata.

> **Important**
>
> This repository does not replace the official MLCommons Training repository.
> It provides a reproducibility and performance-engineering workflow around a
> pinned version of the official MLPerf Training implementation.
>
> Results produced with this repository should be regarded as local
> reproducibility/performance measurements unless all MLCommons submission,
> compliance, system-description, and result-validation requirements are
> separately satisfied.

---

## Supported Reproduction Configurations

| GPU | GPUs | TP | DP | GBS | MBS | Status |
|---|---:|---:|---:|---:|---:|---|
| NVIDIA H200 | 2 | 1 | 2 | 32 | 1 | reproduced |
| NVIDIA H200 | 2 | 2 | 1 | 32 | 1 | reproduced |
| NVIDIA A100 | 8 | 4 | 2 | 32 | 1 | reproduced |

The H200 and A100 experiments reuse the same preprocessed C4 dataset and
the same Llama 3.1 tokenizer.

Experiment-specific NPY/index caches are stored separately.

Additional GPU counts and accelerator generations can be added using the same
repository structure.

---

## Relationship to the Official MLCommons Repository

This repository is a KISTI NEURON reproducibility and system-integration
layer around the official MLCommons Training Small LLM reference implementation.

The main KISTI adaptations include:

- Docker-to-Singularity conversion for KISTI NEURON
- hardware-specific H200 and A100 configurations
- KISTI launch and reproducibility automation
- Python runtime compatibility fixes
- A100-specific FP8 control
- preliminary validation using the last 256 C4 shards before a full-dataset run

For a detailed description of the KISTI changes and their relationship to the
upstream implementation, see:

[Differences from the Official MLCommons Training Repository](docs/differences-from-upstream.md)

A100-specific notes are documented in:

[NVIDIA A100 Reproduction Notes](docs/a100-notes.md)

Exact H200 source-level differences are recorded in:

- `reproducibility/pretrain_llama31_kisti.diff`
- `reproducibility/run_llama31_kisti.diff`

---

## 1. Repository and Working Directory Structure

The recommended working directory is:

```text
$MLPERF_ROOT/
├── kisti-neuron-mlperf-training/
│   ├── configs/
│   │   ├── h200/
│   │   └── a100/
│   ├── scripts/
│   ├── patches/
│   ├── docs/
│   ├── results/
│   ├── containers/
│   └── reproducibility/
├── training/
│   └── small_llm_pretraining/
│       └── nemo/
├── containers/
│   └── mlperf-llama31-h200.sif
├── data/
│   ├── C4_processed/
│   └── npy_indices/
│       ├── h200_2gpu_tp1_dp2/
│       ├── h200_2gpu_tp2_dp1/
│       └── a100_8gpu_tp4_dp2/
├── models/
│   └── Llama-3.1-8B/
├── checkpoints/
└── logs/
```

The KISTI repository contains the NEURON-specific scripts, configurations,
patches, documentation, and result summaries.

The official MLCommons source is cloned separately under:

```text
$MLPERF_ROOT/training
```

Large data, container images, checkpoints, and benchmark logs should remain
outside the Git repository.

---

## 2. Create the Working Directory

Use a dedicated directory:

```bash
export MLPERF_ROOT=/scratch/$USER/mlperf-training-llama31

mkdir -p "$MLPERF_ROOT"
```

Clone this repository:

```bash
git clone \
    https://github.com/hwang2006/kisti-neuron-mlperf-training.git \
    "$MLPERF_ROOT/kisti-neuron-mlperf-training"

cd "$MLPERF_ROOT/kisti-neuron-mlperf-training"
```

Prepare the directory structure:

```bash
./scripts/01_prepare_directories.sh
```

---

## 3. Prepare the Official MLCommons Training Source

This workflow uses the official MLCommons Training repository:

```text
https://github.com/mlcommons/training
```

The tested upstream commit is:

```text
aa344c7fb900e82ed19fb94aebfed50c63ab2204
```

Prepare the pinned source:

```bash
./scripts/02_prepare_mlcommons_source.sh
```

The Small LLM workload should then be available at:

```text
$MLPERF_ROOT/training/small_llm_pretraining/nemo
```

Verify:

```bash
ls -lh \
    "$MLPERF_ROOT/training/small_llm_pretraining/nemo"
```

The KISTI runtime files include the H200 path:

```text
pretrain_llama31_kisti.py
run_llama31_kisti.sh
```

and the A100 path:

```text
pretrain_llama31_a100.py
run_llama31_a100.sh
```

---

## 4. Build the Singularity Container

KISTI NEURON compute nodes use Singularity rather than Docker.

Build the image:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-training"

singularity build --fakeroot \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif" \
    containers/mlperf-h200.def
```

The resulting image should be:

```text
$MLPERF_ROOT/containers/mlperf-llama31-h200.sif
```

Verify:

```bash
ls -lh \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif"
```

Verify major software versions:

```bash
singularity exec --nv \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif" \
    python3 - <<'PY'
import torch
import nemo

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("NeMo:", nemo.__version__)
PY
```

The tested environment includes approximately:

```text
Python              3.12
PyTorch             2.6.0 NVIDIA 25.01 build
CUDA                12.8 inside the container
NeMo                2.1.0
Megatron-Core       0.11.0rc0
Transformer Engine  1.14
```

The host driver may report support for a newer CUDA version. This is normal
because the container carries its own CUDA userspace libraries.

---

## 5. Prepare the C4 Dataset and Tokenizer

### Option A — Download the Preprocessed Dataset

For a new installation:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-training"

./scripts/03_download_preprocessed_data.sh
```

This prepares:

```text
$MLPERF_ROOT/data/C4_processed
```

and:

```text
$MLPERF_ROOT/models/Llama-3.1-8B
```

The preprocessed C4 dataset is large and may require several hundred GB of
storage.

---

### Option B — Reuse an Existing Validated Dataset

If a validated preprocessed C4 dataset and tokenizer already exist, symbolic
links can be used instead of downloading and preprocessing them again.

Example:

```bash
rmdir "$MLPERF_ROOT/data/C4_processed"
rmdir "$MLPERF_ROOT/models/Llama-3.1-8B"

ln -s /path/to/existing/C4_processed \
    "$MLPERF_ROOT/data/C4_processed"

ln -s /path/to/existing/Llama-3.1-8B \
    "$MLPERF_ROOT/models/Llama-3.1-8B"
```

Verify:

```bash
readlink -f "$MLPERF_ROOT/data/C4_processed"
readlink -f "$MLPERF_ROOT/models/Llama-3.1-8B"
```

The same preprocessed dataset and tokenizer can be reused across H200 and A100
runs when the workload, tokenizer, and preprocessing format are unchanged.

---

## 6. NPY / Dataset Index Cache

The preprocessed C4 dataset contains the tokenized data and indexed-dataset
metadata.

At training startup, Megatron/NeMo may also generate NumPy-based document,
sample, and shuffle indices used to map the preprocessed dataset into training
samples.

These caches are not a second copy of the C4 preprocessing output.

Conceptually:

```text
Raw C4
  |
  v
preprocessing
  |
  v
C4_processed
  |-- *.bin
  `-- *.idx
        |
        v
Megatron/NeMo dataset builder
        |
        v
NPY sample/document/shuffle indices
```

The cache directories are separated by experiment to avoid mixing runtime
indices from different configurations.

Examples:

```text
$MLPERF_ROOT/data/npy_indices/h200_2gpu_tp1_dp2
$MLPERF_ROOT/data/npy_indices/h200_2gpu_tp2_dp1
$MLPERF_ROOT/data/npy_indices/a100_8gpu_tp4_dp2
```

The GPU generation itself is not the fundamental reason for separate caches;
the separation is primarily for reproducibility and experiment isolation.

---

## 7. Python Compatibility Fix

The NVIDIA PyTorch container used by this workload includes a `pangu` package
whose API can be incompatible with the NeMo version used by the benchmark.

Without the compatibility fix, startup may fail with:

```text
ImportError: cannot import name 'spacing' from 'pangu'
```

This repository provides:

```text
patches/pythonfix/pangu.py
```

The KISTI launchers prepend this directory to `PYTHONPATH`.

No manual installation of `pangu.py` is required.

---

# H200 Reproduction

## 8. Obtain Two NVIDIA H200 GPUs

Run the H200 reproduction inside a Slurm allocation containing two H200 GPUs.

After entering the allocated compute node:

```bash
hostname
```

Verify the GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
```

Expected configuration:

```text
0, NVIDIA H200, ...
1, NVIDIA H200, ...
```

Before launching, both GPUs should ideally be idle.

---

## 9. H200 TP1 / DP2 Configuration

Configuration:

```text
GPUs               : 2
Tensor Parallelism : 1
Data Parallelism   : 2
Micro Batch Size   : 1
Global Batch Size  : 32
```

Config file:

```text
configs/h200/h200_2gpu_tp1_dp2.sh
```

Run:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-training"

./scripts/run_h200_tp1.sh
```

The wrapper invokes:

```text
run_h200_tp1.sh
    |
    +-- run_h200.sh tp1
            |
            +-- configs/h200/h200_2gpu_tp1_dp2.sh
            |
            +-- Singularity container
            |
            +-- run_llama31_kisti.sh
            |
            +-- pretrain_llama31_kisti.py
```

---

## 10. H200 TP2 / DP1 Configuration

After TP1/DP2 completes, the TP2/DP1 configuration can be used for comparison.

Configuration:

```text
GPUs               : 2
Tensor Parallelism : 2
Data Parallelism   : 1
Micro Batch Size   : 1
Global Batch Size  : 32
```

Config file:

```text
configs/h200/h200_2gpu_tp2_dp1.sh
```

Run:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-training"

./scripts/run_h200_tp2.sh
```

---

## 11. Monitor H200 Runs

Check the training processes:

```bash
ps -ef | grep -E \
    'pretrain_llama31|run_llama31|torchrun|nemo' \
    | grep -v grep
```

Check GPU utilization:

```bash
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
```

Locate recent NeMo combined logs:

```bash
find ~/.nemo_run/experiments \
    -type f \
    -name combined.log \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
    | sort
```

Inspect the most recent training line:

```bash
grep 'Training epoch' <combined.log> | tail -1
```

Inspect validation and completion events:

```bash
grep -E \
    '"key": "eval_accuracy"|"key": "run_stop"' \
    <combined.log> \
    | tail -30
```

---

# A100 Reproduction

## 12. A100 8-GPU TP4 / DP2 Configuration

The reproduced A100 configuration uses:

```text
GPUs               : 8
Tensor Parallelism : 4
Data Parallelism   : 2
Micro Batch Size   : 1
Global Batch Size  : 32
```

Config file:

```text
configs/a100/a100_8gpu_tp4_dp2.sh
```

Important A100-specific settings:

```bash
export DISABLE_FP8=1
export ENABLE_RECOMPUTE=0
export DISABLE_CE_FUSION=0
```

`DISABLE_FP8=1` selects the A100-compatible BF16 path instead of the FP8
configuration used by the H200-oriented reference path.

The final reproduced TP4/DP2 configuration keeps cross-entropy fusion enabled.

Historical debugging code also provides optional controls for activation
recomputation and disabling cross-entropy fusion, but these are not enabled in
the final A100 TP4/DP2 reproduction.

---

## 13. Run the A100 Reproduction

Obtain an allocation with eight NVIDIA A100 GPUs and verify:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
```

Then:

```bash
export MLPERF_ROOT=/scratch/$USER/mlperf-training-llama31

cd "$MLPERF_ROOT/kisti-neuron-mlperf-training"

./scripts/run_a100_tp4_dp2.sh
```

The wrapper uses:

```text
run_a100_tp4_dp2.sh
    |
    +-- run_a100.sh tp4_dp2
            |
            +-- configs/a100/a100_8gpu_tp4_dp2.sh
            |
            +-- Singularity container
            |
            +-- run_llama31_a100.sh
            |
            +-- pretrain_llama31_a100.py
```

A100-specific notes are available in:

```text
docs/a100-notes.md
```

---

## 14. Historical A100 Result

The historical KISTI A100 reproduction produced:

| Item | Value |
|---|---:|
| GPUs | 8 x NVIDIA A100 |
| TP | 4 |
| DP | 2 |
| GBS | 32 |
| MBS | 1 |
| Seed | 12875 |
| Final evaluation value | 3.2832148075 |
| Samples at final evaluation | 233440 |
| Train samples at run stop | 233472 |
| Status | success |

This result is included as a reproducibility reference and is not presented as
an official MLPerf submission.

See:

```text
results/a100/8gpu_tp4_dp2/README.md
```

---

## 15. Dataset Selection and Preliminary Validation

The current KISTI reproduction path uses:

```text
--use_last_256_shards
```

for **preliminary validation before a full-dataset benchmark run**.

This option was introduced to verify that the complete workflow executes
correctly before committing substantial resources to a full-dataset run.

The consolidated C4 shards are organized approximately as:

```text
c4-train.en_0  <- raw shards   0-127
c4-train.en_1  <- raw shards 128-255
c4-train.en_2  <- raw shards 256-383
c4-train.en_3  <- raw shards 384-511
c4-train.en_4  <- raw shards 512-639
c4-train.en_5  <- raw shards 640-767
c4-train.en_6  <- raw shards 768-895
c4-train.en_7  <- raw shards 896-1023
```

The preliminary mode uses:

```text
c4-train.en_6
c4-train.en_7
```

and the validation subset:

```text
c4-validation-91205-samples.en
```

This shard selection should not be interpreted as redefining the official
MLPerf Training dataset.

A future full-dataset run should remove or otherwise disable the preliminary
last-256-shard selection and be validated separately.

---

## 16. Verify Benchmark Inputs

Verify the container:

```bash
ls -lh \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif"
```

Verify the preprocessed C4 dataset:

```bash
du -shL \
    "$MLPERF_ROOT/data/C4_processed"
```

Verify the training shards used by preliminary validation:

```bash
ls -lh \
    "$MLPERF_ROOT/data/C4_processed/c4-train.en_6_text_document.bin" \
    "$MLPERF_ROOT/data/C4_processed/c4-train.en_6_text_document.idx" \
    "$MLPERF_ROOT/data/C4_processed/c4-train.en_7_text_document.bin" \
    "$MLPERF_ROOT/data/C4_processed/c4-train.en_7_text_document.idx"
```

Verify the validation data:

```bash
ls -lh \
    "$MLPERF_ROOT/data/C4_processed/c4-validation-91205-samples.en_text_document.bin" \
    "$MLPERF_ROOT/data/C4_processed/c4-validation-91205-samples.en_text_document.idx"
```

Verify the tokenizer:

```bash
ls -lh \
    "$MLPERF_ROOT/models/Llama-3.1-8B/tokenizer.json" \
    "$MLPERF_ROOT/models/Llama-3.1-8B/tokenizer_config.json"
```

---

## 17. Result Validation

The success criterion used by this workload is:

```text
target_log_ppl <= 3.3
```

The MLPerf log may use the key:

```text
eval_accuracy
```

for the reported numerical validation value.

A successful run should contain an MLPerf `run_stop` event with:

```text
status: success
```

Search a detailed combined log with:

```bash
grep -E \
    '"key": "eval_accuracy"|"key": "run_stop"' \
    <combined.log>
```

---

## 18. Troubleshooting

### `ImportError: cannot import name 'spacing' from 'pangu'`

Verify:

```bash
ls -lh \
    "$MLPERF_ROOT/kisti-neuron-mlperf-training/patches/pythonfix/pangu.py"
```

The launcher should prepend the compatibility directory to `PYTHONPATH`.

---

### `du -sh C4_processed` reports `0`

If `C4_processed` is a symbolic link, use:

```bash
du -shL \
    "$MLPERF_ROOT/data/C4_processed"
```

---

### Training shard files cannot be found

Incorrect:

```text
c4-train.en_6.bin
c4-train.en_6.idx
```

Correct:

```text
c4-train.en_6_text_document.bin
c4-train.en_6_text_document.idx
```

Validation files similarly use the `_text_document` suffix.

---

### Launcher starts but immediately exits

Inspect the launcher log and check:

```bash
nvidia-smi
```

and:

```bash
ps -ef | grep -E \
    'pretrain_llama31|run_llama31|torchrun|nemo' \
    | grep -v grep
```

Also search logs for:

```text
Traceback
CUDA out of memory
RuntimeError
ImportError
Killed
```

---

## 19. Reproducibility Notes

The MLCommons Training source is pinned to:

```text
aa344c7fb900e82ed19fb94aebfed50c63ab2204
```

The following large artifacts should not be committed to Git:

```text
C4 dataset
preprocessed C4 data
Singularity SIF images
model weights
checkpoints
generated NPY/index files
large benchmark logs
```

The repository should contain only scripts, configurations, patches,
documentation, provenance information, small result summaries, and other
metadata required to reconstruct the environment.

---

## 20. Quick Start

### Common preparation

```bash
export MLPERF_ROOT=/scratch/$USER/mlperf-training-llama31

mkdir -p "$MLPERF_ROOT"

git clone \
    https://github.com/hwang2006/kisti-neuron-mlperf-training.git \
    "$MLPERF_ROOT/kisti-neuron-mlperf-training"

cd "$MLPERF_ROOT/kisti-neuron-mlperf-training"

./scripts/01_prepare_directories.sh
./scripts/02_prepare_mlcommons_source.sh
./scripts/03_download_preprocessed_data.sh

singularity build --fakeroot \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif" \
    containers/mlperf-h200.def
```

### H200 — 2 GPUs, TP1 / DP2

```bash
./scripts/run_h200_tp1.sh
```

### H200 — 2 GPUs, TP2 / DP1

```bash
./scripts/run_h200_tp2.sh
```

### A100 — 8 GPUs, TP4 / DP2

```bash
./scripts/run_a100_tp4_dp2.sh
```

---

## References

- MLCommons Training
  https://github.com/mlcommons/training

- MLPerf Training
  https://mlcommons.org/benchmarks/training/

- NVIDIA NeMo
  https://github.com/NVIDIA/NeMo

- KISTI National Supercomputing Center
  https://www.ksc.re.kr/

---

## License

This repository contains KISTI-specific scripts, configurations,
documentation, patches, and reproducibility metadata.

The upstream MLCommons, NVIDIA NeMo, PyTorch, and other third-party components
remain subject to their respective licenses.
