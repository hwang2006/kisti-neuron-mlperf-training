# MLPerf Training Reproduction Results

## Common Notes

These results are local KISTI NEURON reproduction and performance-engineering
measurements based on the MLCommons Training Small LLM / Llama 3.1 8B workload.

Common parameters:

- Model: Llama 3.1 8B
- Sequence length: 8192
- Global batch size: 32
- Micro batch size: 1
- Maximum learning rate: 5e-4
- Warmup steps: 512
- Target validation log perplexity: 3.3
- Checkpoint saving: disabled

The historical results documented below were obtained with the preliminary
`--use_last_256_shards` mode and are retained for reproducibility.

The current benchmark configurations default to the full preprocessed C4
training dataset (`c4-train.en_0` through `c4-train.en_7`). Full-dataset
results will be recorded separately after successful completion.

---

## H200 Results

### Current Clean TP1 / DP2 Rerun

Configuration:

```text
GPUs = 2 x NVIDIA H200
TP   = 1
DP   = 2
MBS  = 1
GBS  = 32
Seed = 25646
```

MLPerf events:

```text
run_start = 1787844598029 ms
```

Final successful evaluation:

```text
eval_accuracy = 3.289571523666382
samples_count = 208864
```

Run stop:

```text
run_stop = 1787902805703 ms
status   = success
```

Elapsed time:

```text
58207.674 seconds
16 h 10 m 7.674 s
```

A compact result record is stored in:

```text
results/h200/2gpu_tp1_dp2/README.md
```

---

### Historical TP1 / DP2 Reference

Final successful evaluation:

```text
eval_accuracy = 3.2801718711853027
samples_count = 208864
```

Elapsed time:

```text
58614.215 seconds
16 h 16 m 54.215 s
```

---

### Historical TP2 / DP1 Reference

Final successful evaluation:

```text
eval_accuracy = 3.2964823246002197
samples_count = 208864
```

Elapsed time:

```text
58425.207 seconds
16 h 13 m 45.207 s
```

---

### H200 Comparison

| Run | Final validation log-ppl | Samples to target | Time |
|---|---:|---:|---:|
| Current TP1 / DP2, seed 25646 | 3.2895715237 | 208864 | 16:10:07.674 |
| Historical TP1 / DP2 | 3.2801718712 | 208864 | 16:16:54.215 |
| Historical TP2 / DP1 | 3.2964823246 | 208864 | 16:13:45.207 |

The current clean TP1/DP2 rerun reached the target at the same sample count as
the two historical H200 runs.

Small differences between runs should be treated as observed single-run
variation rather than proof that one parallelization strategy is inherently
faster.

---

## A100 Results

### Historical A100 8-GPU TP4 / DP2

Configuration:

```text
GPUs = 8 x NVIDIA A100
TP   = 4
DP   = 2
MBS  = 1
GBS  = 32
Seed = 12875
```

Important runtime settings:

```bash
DISABLE_FP8=1
ENABLE_RECOMPUTE=0
DISABLE_CE_FUSION=0
```

Final successful evaluation:

```text
eval_accuracy = 3.2832148075
samples_count = 233440
train_samples = 233472
status        = success
```

A compact result record is stored in:

```text
results/a100/8gpu_tp4_dp2/README.md
```

This historical A100 run also used `--use_last_256_shards`.

---

## Interpretation of `eval_accuracy`

The MLPerf log key is named `eval_accuracy`, but for this workload the
reported numerical value corresponds to the validation log-perplexity/loss
quality metric.

Lower values are better.

The target used by this workflow is:

```text
target_log_ppl <= 3.3
```

All successful runs listed above satisfied the target.

---

## Dataset Scope

The current KISTI benchmark workflow defaults to the full preprocessed C4
training dataset (`c4-train.en_0` through `c4-train.en_7`).

The historical results above used `--use_last_256_shards` for preliminary
end-to-end validation.

This selects the consolidated C4 datasets corresponding to the final 256 raw
training shards:

```text
c4-train.en_6
c4-train.en_7
```

This mode was used to validate the execution pipeline before the full-dataset
benchmark configuration became the default. It should not be interpreted as
redefining the official MLPerf Training dataset.

---

## Submission Status

These runs use MLPerf logging and the MLCommons workload implementation, but
this repository does not claim that the results are official MLCommons
submissions.

A formal submission would additionally require compliance with the applicable
MLPerf Training rules, system-description requirements, compliance procedures,
result packaging, and MLCommons review.
