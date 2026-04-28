# Rotate + BFP Experiment Stub

This directory is a small, standalone subset of SpinKV for running HF causal LM
perplexity with only rotation and BFP options.

## Files

- `run_model.py`: entrypoint for loading a model and evaluating WikiText-2 PPL
- `utils.py`: BFP quantization helpers and PPL evaluation
- `hadamard_utils.py`: Hadamard matrix helpers
- `rotquant/`: norm fusion, rotation, BFP, and attention monkey patches

## Example

```bash
python run_model.py \
  --model facebook/opt-1.3b \
  --device cuda \
  --rotate \
  --online_rotate \
  --bfp \
  --bfp_bits 8 \
  --bfp_block_size 128
```

`--rotate` always applies Hadamard rotation. `--online_rotate` enables both
post-RoPE Q/K Hadamard rotation and online MLP/FFN intermediate rotation.
