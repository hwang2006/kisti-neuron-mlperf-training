# H200 Benchmark Results

## Configuration

The successful experiments used a single KISTI NEURON node with two
NVIDIA H200 GPUs.

Common parameters:

- Model: Llama 3.1 8B
- Sequence length: 8192
- Global batch size: 32
- Micro batch size: 1
- Maximum learning rate: 5e-4
- Warmup steps: 512
- Target validation log perplexity: 3.3
- Checkpoint saving: disabled

## TP1 / DP2

Configuration:

    tensor parallelism = 1
    data parallelism   = 2
    micro batch size   = 1
    global batch size  = 32

MLPerf events:

    run_start = 1786896330437 ms

Final successful evaluation:

    eval_accuracy = 3.2801718711853027
    samples_count = 208864

    run_stop = 1786954944652 ms
    status   = success

Elapsed time:

    58614.215 seconds
    16 h 16 m 54.215 s

## TP2 / DP1

Configuration:

    tensor parallelism = 2
    data parallelism   = 1
    micro batch size   = 1
    global batch size  = 32

MLPerf events:

    run_start = 1786965611445 ms

Final successful evaluation:

    eval_accuracy = 3.2964823246002197
    samples_count = 208864

    run_stop = 1787024036652 ms
    status   = success

Elapsed time:

    58425.207 seconds
    16 h 13 m 45.207 s

## Comparison

| Run | Final validation log-ppl | Samples to target | Time |
| --- | ---: | ---: | ---: |
| TP1 / DP2 | 3.2801718712 | 208864 | 16:16:54.215 |
| TP2 / DP1 | 3.2964823246 | 208864 | 16:13:45.207 |

Difference:

    189.008 seconds

TP2 was approximately:

    0.32%

faster than TP1 in this particular pair of experiments.

This difference is small and the runs used different random seeds.
Therefore it should be treated as an observed single-run difference,
not as proof that TP2 is inherently faster.

## Interpretation of eval_accuracy

The MLPerf log key is named `eval_accuracy`, but for this workload the
reported value represents the validation log-perplexity/loss quality
metric.

Lower values are better.

The target is:

    validation log perplexity <= 3.3

Both runs satisfied the target.

## Submission status

These runs use MLPerf logging and the MLCommons workload implementation,
but the repository does not claim that these results are official
MLCommons submissions.

Submission metadata, compliance procedures, and the official result
packaging process would need to be completed separately for a formal
MLPerf submission.
