# Architecture

## Overview

The project is organized around a small set of stable layers:

1. Data layer: loads CLINC150, delegates text processing to a tokenizer
   backend, builds tokenizer state, and converts examples into padded tensors
   with sequence lengths.
2. Model layer: owns the embedding table, encoder backend, and classifier head.
3. Training layer: builds dataloaders, runs optimization, evaluates splits, and
   writes checkpoints.
4. Inference layer: reloads the checkpoint, reconstructs the model from stored
   config, and predicts labels for free-form text.

## Tokenization Composition

Tokenizer backends live in `data/tokenizers.py` and expose a stable runtime
interface:

1. `prepare()` fetches backend resources.
2. `fit()` learns tokenizer state from training text when required.
3. `tokenize()` supports dataset inspection and statistics.
4. `encode()` converts text into the integer ids expected by the model.

The default backend is `nltk_word`, which lowercases, filters punctuation-only
tokens, and builds a simple token-to-id vocabulary. Future backends such as BPE
should implement the same interface and keep the rest of the pipeline unchanged.

The current tokenizer backends are:

- `nltk_word`: NLTK word tokenization plus lowercase and punctuation filtering.
- `hf_bpe`: a Hugging Face `tokenizers` BPE model trained from the full CLINC
  train split and serialized into the checkpoint.

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
- `tokenizer_state`
- `vocab` for backward compatibility with older checkpoints
- serialized experiment config

The config payload and tokenizer state are the source of truth for
reconstructing the input pipeline and the model. This is what makes backend
switching and future extensions safe.

## Extension Pattern

To add a new encoder:

1. Implement a new encoder module in `models/encoders.py`.
2. Register it in `build_encoder`.
3. Extend `ModelConfig` if the backend needs new hyperparameters.
4. Document the new backend in `README.md`, `AGENTS.md`, and this file.

The training and inference code should not need structural changes if the new
encoder follows the existing interface.

To add a new tokenizer backend:

1. Implement a tokenizer class in `data/tokenizers.py`.
2. Register it in `build_tokenizer`.
3. Extend `TokenizerConfig` if the backend needs new settings.
4. Keep `IntentDataset`, training, and inference operating only on the shared
   tokenizer interface.
