# KISTI NEURON MLPerf Training Reproduction — Llama 3.1 8B on NVIDIA GPUs

This repository provides a reproducible workflow for running the
**MLPerf Training Small LLM / Llama 3.1 8B benchmark** on the
KISTI NEURON GPU cluster using NVIDIA A100 and H200 GPUs.

The workflow is based on the official MLCommons Training repository and adds
KISTI NEURON-specific configuration, runtime fixes, and launch scripts.

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

| GPU | GPUs | TP | DP | Status |
|---|---:|---:|---:|---|
| NVIDIA H200 | 2 | 1 | 2 | reproduced |
| NVIDIA H200 | 2 | 2 | 1 | reproduced |
| NVIDIA A100 | 8 | 4 | 2 | reproduced |

The same preprocessed C4 dataset and Llama 3.1 tokenizer are reused across
the H200 and A100 experiments. Experiment-specific NPY/index caches are kept
separate.

## Relationship to the Official MLCommons Repository

This repository is a KISTI NEURON reproducibility and system-integration
layer around the official MLCommons Training Small LLM reference implementation.

The main KISTI adaptations include:

- Docker-to-Singularity conversion for KISTI NEURON
- NVIDIA H200 TP1/DP2 and TP2/DP1 configurations
- NVIDIA A100 8-GPU TP4/DP2 configuration
- KISTI launch and reproducibility automation
- Python runtime compatibility fixes
- preliminary validation using the last 256 C4 shards before a full-dataset run

For a detailed description of what was changed, why it was changed, and how
the KISTI workflow relates to the official reference, see:

[Differences from the Official MLCommons Training Repository](docs/differences-from-upstream.md)

Exact source-level differences are recorded in:

- `reproducibility/pretrain_llama31_kisti.diff`
- `reproducibility/run_llama31_kisti.diff`

---

## 1. Repository Structure

The recommended working directory is:

```text
$MLPERF_ROOT/
├── kisti-neuron-mlperf-llama31-training/
├── training/
│   └── small_llm_pretraining/
│       └── nemo/
├── containers/
│   └── mlperf-llama31-h200.sif
├── data/
│   ├── C4_processed/
│   └── npy_indices_h200_tp1/
├── models/
│   └── Llama-3.1-8B/
├── checkpoints/
│   └── continual/
└── logs/
    ├── mlperf-h200-tp1/
    └── mlperf-h200-tp2/
```

The KISTI repository contains the NEURON-specific scripts and configuration.
The official MLCommons source is cloned separately under:

```text
$MLPERF_ROOT/training
```

---

## 2. Create the Working Directory

Use a dedicated directory for this benchmark.

```bash
export MLPERF_ROOT=/scratch/$USER/mlperf-training-llama31

mkdir -p "$MLPERF_ROOT"
```

Clone this repository:

```bash
git clone \
    https://github.com/hwang2006/kisti-neuron-mlperf-llama31-training.git \
    "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

cd "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"
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

The MLPerf Small LLM workload should then be available at:

```text
$MLPERF_ROOT/training/small_llm_pretraining/nemo
```

Verify:

```bash
ls -lh \
    "$MLPERF_ROOT/training/small_llm_pretraining/nemo"
```

The KISTI-specific runtime files should include:

```text
pretrain_llama31_kisti.py
run_llama31_kisti.sh
```

---

## 4. Build the Singularity Container

The benchmark uses a Singularity image based on the NVIDIA PyTorch container
required by the MLPerf Small LLM workload.

Build the image:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

singularity build --fakeroot \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif" \
    containers/mlperf-h200.def
```

The resulting image should be:

```text
$MLPERF_ROOT/containers/mlperf-llama31-h200.sif
```

Verify the image:

```bash
ls -lh \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif"
```

Verify the major software versions:

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

The host NVIDIA driver may report support for a newer CUDA version. This is
normal because the container carries its own CUDA userspace libraries.

---

## 5. Prepare the C4 Dataset and Tokenizer

### Option A — Download the Preprocessed Dataset

For a new installation, use:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

./scripts/03_download_preprocessed_data.sh
```

This prepares the preprocessed C4 dataset under:

```text
$MLPERF_ROOT/data/C4_processed
```

and the Llama 3.1 tokenizer under:

```text
$MLPERF_ROOT/models/Llama-3.1-8B
```

The preprocessed C4 dataset is very large and may require several hundred GB
of storage.

---

### Option B — Reuse an Existing Validated Dataset

If a validated preprocessed C4 dataset and tokenizer already exist on the
system, symbolic links can be used instead of downloading them again.

For example:

```bash
rmdir "$MLPERF_ROOT/data/C4_processed"
rmdir "$MLPERF_ROOT/models/Llama-3.1-8B"

ln -s /path/to/existing/C4_processed \
    "$MLPERF_ROOT/data/C4_processed"

ln -s /path/to/existing/Llama-3.1-8B \
    "$MLPERF_ROOT/models/Llama-3.1-8B"
```

Verify the links:

```bash
readlink -f "$MLPERF_ROOT/data/C4_processed"
readlink -f "$MLPERF_ROOT/models/Llama-3.1-8B"
```

---

## 6. Python Compatibility Fix

The NVIDIA PyTorch container used by this benchmark includes a `pangu`
package whose API is incompatible with the NeMo version used by this MLPerf
workload.

Without the compatibility fix, NeMo may fail during startup with an error such
as:

```text
ImportError: cannot import name 'spacing' from 'pangu'
```

This repository provides a compatible implementation at:

```text
patches/pythonfix/pangu.py
```

The H200 launcher automatically prepends this directory to `PYTHONPATH`:

```text
$REPO_ROOT/patches/pythonfix
```

where `REPO_ROOT` is the root directory of this Git repository.

No manual installation or copying of `pangu.py` is required.

The launcher verifies that the compatibility file exists before starting the
benchmark.

You can test the compatibility fix manually:

```bash
export REPO="$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

singularity exec --nv \
    -B /scratch:/scratch \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif" \
    /bin/bash -lc "
        export PYTHONPATH=$REPO/patches/pythonfix:\$PYTHONPATH

        python3 - <<'PY'
import pangu

print("pangu file:", pangu.__file__)
print("has spacing:", hasattr(pangu, "spacing"))
print("spacing test:", pangu.spacing("Hello世界"))
PY
    "
```

Expected output includes:

```text
has spacing: True
spacing test: Hello 世界
```

---

## 7. Obtain Two NVIDIA H200 GPUs

Run the benchmark inside a Slurm allocation containing two NVIDIA H200 GPUs.

After entering the allocated compute node, verify the hostname:

```bash
hostname
```

Verify the GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
```

The expected configuration is similar to:

```text
0, NVIDIA H200, 143771 MiB, <driver-version>
1, NVIDIA H200, 143771 MiB, <driver-version>
```

Before launching a benchmark, both GPUs should ideally be idle.

Check:

```bash
nvidia-smi \
    --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
```

---

## 8. Verify the Benchmark Inputs

Before launching the benchmark, verify the container:

```bash
ls -lh \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif"
```

Verify the preprocessed C4 dataset:

```bash
du -shL \
    "$MLPERF_ROOT/data/C4_processed"
```

The `-L` option follows symbolic links if the dataset is reused from another
location.

NeMo/Megatron preprocessing generates files using the
`_text_document.bin` and `_text_document.idx` suffixes.

Verify the required training shards:

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

Verify the KISTI runtime files and Python compatibility fix:

```bash
ls -lh \
    "$MLPERF_ROOT/training/small_llm_pretraining/nemo/pretrain_llama31_kisti.py" \
    "$MLPERF_ROOT/training/small_llm_pretraining/nemo/run_llama31_kisti.sh" \
    "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training/patches/pythonfix/pangu.py"
```

Verify that two H200 GPUs are visible:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
```

The expected configuration is two visible NVIDIA H200 GPUs.

---

## 9. TP1 / DP2 Configuration

The first benchmark configuration uses:

```text
Tensor Parallelism : TP = 1
Data Parallelism   : DP = 2
GPUs               : 2
Micro Batch Size   : 1
Global Batch Size  : 32
```

The configuration file is:

```text
configs/h200/h200_2gpu_tp1_dp2.sh
```

Important settings include:

```bash
export GBS=32
export MBS=1
export MAX_STEPS=1200000
export WARMUP_STEPS=512
export MAX_LR="5e-4"
export EVAL_EVERY=12288
export SAVE_CKPT=0
export USE_CKPT=0
```

Inspect the configuration before running:

```bash
grep -E \
    '^(export )?(TP|GBS|MBS|MAX_LR|MAX_STEPS|WARMUP_STEPS|EVAL_EVERY|SAVE_CKPT|USE_CKPT)=' \
    configs/h200/h200_2gpu_tp1_dp2.sh
```

---

## 10. Run TP1 / DP2

Start the benchmark:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

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

The launcher uses `nohup` and starts the benchmark in the background.

Typical output is:

```text
===== MLPerf Training Launch =====
Mode             : tp1
MLPERF_ROOT      : ...
Training source  : ...
Container        : ...
TP               : 1
GPUs             : 2
MBS / GBS        : 1 / 32
Job directory    : ...
Launcher log     : ...
PID=<pid>
LOG=<log-file>
```

The shell can therefore return immediately while the training process
continues in the background.

> `nohup` protects the process from an SSH/session disconnect, but it does not
> extend the Slurm allocation. The Slurm job must remain active for the entire
> benchmark.

---

## 11. Monitor the TP1 Run

Check the launcher process:

```bash
ps -ef | grep -E \
    'pretrain_llama31|run_llama31|torchrun|nemo' \
    | grep -v grep
```

Check GPU utilization:

```bash
nvidia-smi
```

or:

```bash
nvidia-smi \
    --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
```

List launcher logs:

```bash
find "$MLPERF_ROOT/logs/mlperf-h200-tp1" \
    -maxdepth 1 \
    -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
    | sort
```

Inspect the latest launcher log:

```bash
tail -n 100 \
    "$MLPERF_ROOT/logs/mlperf-h200-tp1/launcher_<timestamp>.log"
```

Follow it interactively:

```bash
tail -f \
    "$MLPERF_ROOT/logs/mlperf-h200-tp1/launcher_<timestamp>.log"
```

Pressing `Ctrl-C` stops only `tail -f`; it does not stop the background
training process.

During initialization, verify that there are no errors such as:

```text
Traceback
CUDA out of memory
RuntimeError
ImportError
Killed
```

---

## 12. TP2 / DP1 Configuration

After completing the TP1/DP2 measurement, the second configuration can be
used for comparison.

```text
Tensor Parallelism : TP = 2
Data Parallelism   : DP = 1
GPUs               : 2
Micro Batch Size   : 1
Global Batch Size  : 32
```

The configuration file is:

```text
configs/h200/h200_2gpu_tp2_dp1.sh
```

Run:

```bash
cd "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

./scripts/run_h200_tp2.sh
```

Logs are written under:

```text
$MLPERF_ROOT/logs/mlperf-h200-tp2
```

---

## 13. Dataset Selection

The KISTI runtime uses:

```text
--use_last_256_shards
```

for the benchmark training dataset.

The consolidated preprocessed C4 shards are organized approximately as:

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

The benchmark uses:

```text
c4-train.en_6
c4-train.en_7
```

and the validation subset:

```text
c4-validation-91205-samples.en
```

The actual Megatron dataset files include the `_text_document` suffix.

---

## 14. Result Validation

The benchmark success criterion used by this workflow is based on the MLPerf
Small LLM target validation value:

```text
target_log_ppl <= 3.3
```

The MLPerf log may use the key:

```text
eval_accuracy
```

for the reported validation metric even though the numerical value corresponds
to the workload's validation log-perplexity/loss criterion.

A successful run should contain an MLPerf `run_stop` event with success status.

Search the logs for important MLPerf events:

```bash
grep -E \
    'run_start|eval_accuracy|run_stop' \
    "$MLPERF_ROOT"/logs/mlperf-h200-tp1/* \
    2>/dev/null
```

The same can be done for TP2:

```bash
grep -E \
    'run_start|eval_accuracy|run_stop' \
    "$MLPERF_ROOT"/logs/mlperf-h200-tp2/* \
    2>/dev/null
```

Depending on NeMo Run, detailed combined logs may also be stored under:

```text
~/.nemo_run/experiments/
```

Recent combined logs can be located with:

```bash
find ~/.nemo_run/experiments \
    -type f \
    -name combined.log \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
    | sort
```

---

## 15. Troubleshooting

### `ImportError: cannot import name 'spacing' from 'pangu'`

Example:

```text
ImportError: cannot import name 'spacing' from 'pangu'
```

Confirm that the repository compatibility file exists:

```bash
ls -lh \
    "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training/patches/pythonfix/pangu.py"
```

Confirm that the launcher uses it:

```bash
grep -nE \
    'PYTHONFIX|PYTHONPATH|pangu' \
    "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training/scripts/run_h200.sh"
```

The launcher should prepend:

```text
$REPO_ROOT/patches/pythonfix
```

to `PYTHONPATH`.

---

### `du -sh C4_processed` reports `0`

If `C4_processed` is a symbolic link, this is expected.

Use:

```bash
du -shL \
    "$MLPERF_ROOT/data/C4_processed"
```

to follow the symbolic link.

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

Likewise, the validation files are:

```text
c4-validation-91205-samples.en_text_document.bin
c4-validation-91205-samples.en_text_document.idx
```

---

### Launcher starts but immediately exits

First identify the launcher log:

```bash
find "$MLPERF_ROOT/logs" \
    -type f \
    -name 'launcher_*.log' \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
    | sort
```

Then inspect it:

```bash
tail -n 200 <launcher-log>
```

Also verify:

```bash
nvidia-smi
```

and:

```bash
ps -ef | grep -E \
    'pretrain_llama31|run_llama31|torchrun|nemo' \
    | grep -v grep
```

---

## 16. Reproducibility Notes

For reproducibility, this workflow pins the MLCommons Training source to:

```text
aa344c7fb900e82ed19fb94aebfed50c63ab2204
```

The following large artifacts should not be committed to Git:

```text
C4 dataset
preprocessed C4 data
Singularity SIF images
Llama model weights
checkpoints
generated index files
large benchmark logs
```

The repository should contain only the scripts, configurations, patches,
documentation, provenance information, and small reproducibility metadata
required to reconstruct the environment.

---

## 17. Quick Start Summary

```bash
# 1. Working directory
export MLPERF_ROOT=/scratch/$USER/mlperf-training-llama31
mkdir -p "$MLPERF_ROOT"

# 2. Clone KISTI reproduction repository
git clone \
    https://github.com/hwang2006/kisti-neuron-mlperf-llama31-training.git \
    "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

cd "$MLPERF_ROOT/kisti-neuron-mlperf-llama31-training"

# 3. Prepare directories
./scripts/01_prepare_directories.sh

# 4. Prepare pinned MLCommons source
./scripts/02_prepare_mlcommons_source.sh

# 5. Prepare dataset/tokenizer
./scripts/03_download_preprocessed_data.sh

# 6. Build container
singularity build --fakeroot \
    "$MLPERF_ROOT/containers/mlperf-llama31-h200.sif" \
    containers/mlperf-h200.def

# 7. Verify GPUs
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader

# 8. Run TP1 / DP2
./scripts/run_h200_tp1.sh

# 9. After TP1 completes, run TP2 / DP1
./scripts/run_h200_tp2.sh
```

---

## References

- MLCommons Training  
  https://github.com/mlcommons/training

- MLPerf  
  https://mlcommons.org/benchmarks/training/

- NVIDIA NeMo  
  https://github.com/NVIDIA/NeMo

- KISTI National Supercomputing Center  
  https://www.ksc.re.kr/

---

## License

This repository contains KISTI-specific scripts, configuration, documentation,
and patches for reproducing the experiment.

The upstream MLCommons, NVIDIA NeMo, PyTorch, and other third-party components
remain subject to their respective licenses.
