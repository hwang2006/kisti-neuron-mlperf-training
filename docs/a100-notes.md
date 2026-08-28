# NVIDIA A100 Reproduction Notes

## Configuration

- GPUs: 8 x NVIDIA A100
- TP: 4
- DP: 2
- GBS: 32
- MBS: 1
- FP8: disabled
- Activation recomputation: disabled
- Cross-entropy fusion: enabled

The A100 and H200 experiments reuse the same preprocessed C4 dataset
and the same Llama 3.1 tokenizer.

Each experiment uses a separate NPY/index cache directory.

## Historical Result

- Seed: 12875
- Final evaluation: 3.2832148075
- Samples at final evaluation: 233440
- Train samples: 233472
- Status: success

## Dataset Scope

The historical A100 runner used `--use_last_256_shards`.

Therefore this result is a preliminary/reproduction result,
not an official full-dataset MLPerf submission.
