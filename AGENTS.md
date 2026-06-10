# Agent Context

## Purpose

This repository hosts a neural intent classification project centered on the
CLINC150 dataset. The current production path is a mean-pooled embedding model,
but the codebase is intentionally structured so alternate sequence encoders can
plug into the same classifier and training pipeline. Tokenization is also
expected to remain pluggable so alternate text encoders such as BPE can reuse
the same dataset, training, and checkpoint flow.

## Primary Design Rules

1. Keep encoder backends interchangeable through the shared classifier
   interface in `neural_intent_classification/models/`.
2. Keep tokenization backends interchangeable through the shared tokenizer
   interface in `neural_intent_classification/data/tokenizers.py`.
3. Avoid baking model-specific assumptions into dataset code, training code, or
   checkpoint loading.
4. Preserve a checkpoint format that captures enough config and tokenizer state
   to restore either `mean_pool` or `lstm` models without manual edits.
5. Treat the root scripts (`train.py`, `predict.py`, `explore.py`) as thin
   entrypoints only. Reusable logic belongs in the package.

## Important Paths

- `neural_intent_classification/config.py`: experiment, dataset, tokenizer,
  model, and training configuration dataclasses.
- `neural_intent_classification/data/tokenizers.py`: tokenizer backend
  registry, resource setup, fitting, serialization, and text-to-id encoding.
- `neural_intent_classification/data/`: loading, tokenization, vocab, dataset.
- `neural_intent_classification/models/encoders.py`: encoder backend registry.
- `neural_intent_classification/models/classifier.py`: shared classifier head.
- `neural_intent_classification/training/trainer.py`: training loop,
  evaluation, and checkpointing.
- `neural_intent_classification/inference.py`: checkpoint restore and
  prediction helpers.
- `docs/architecture.md`: high-level system design.
- `docs/development.md`: commands and extension workflow.

## Working Conventions

- Default backend is `mean_pool`.
- Default tokenizer backend is `nltk_word`.
- The supported learned tokenizer backend is `hf_bpe`, trained from the CLINC
  train split and restored from checkpoint state.
- New backends should expose a stable output dimension so the classifier head
  does not need training-script changes.
- If a new encoder needs extra sequence metadata, add it to the dataset output
  in a model-agnostic way. `lengths` is acceptable; encoder-specific tensors are
  not.
- If a new tokenizer needs learned state, serialize it through the tokenizer
  checkpoint payload instead of teaching the training or inference code about
  backend-specific files.
- Do not commit model checkpoints or logs.
- Keep documentation updated when adding encoders, changing checkpoint schema,
  or modifying scripts.
