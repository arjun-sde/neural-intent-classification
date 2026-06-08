# Architecture

## Overview

The project is organized around a small set of stable layers:

1. Data layer: loads CLINC150, tokenizes text, builds the vocabulary, and
   converts examples into padded tensors with sequence lengths.
2. Model layer: owns the embedding table, encoder backend, and classifier head.
3. Training layer: builds dataloaders, runs optimization, evaluates splits, and
   writes checkpoints.
4. Inference layer: reloads the checkpoint, reconstructs the model from stored
   config, and predicts labels for free-form text.

## Model Composition

`IntentClassifier` has three responsibilities:

1. Embed token ids.
2. Delegate sentence representation to the selected encoder backend.
3. Map the encoded representation to intent logits via a small feed-forward
   head.

Today the supported backends are:

- `mean_pool`: masks pad tokens, averages token embeddings, then applies the
  classifier head.
- `lstm`: runs packed sequences through an LSTM and uses the final hidden state
  as the sentence representation before the same classifier head.

This keeps the training loop backend-agnostic.

## Checkpoint Contract

Checkpoints store:

- `model_state_dict`
- `vocab`
- serialized experiment config

The config payload is the source of truth for reconstructing the model. This is
what makes backend switching and future extensions safe.

## Extension Pattern

To add a new encoder:

1. Implement a new encoder module in `models/encoders.py`.
2. Register it in `build_encoder`.
3. Extend `ModelConfig` if the backend needs new hyperparameters.
4. Document the new backend in `README.md`, `AGENTS.md`, and this file.

The training and inference code should not need structural changes if the new
encoder follows the existing interface.
