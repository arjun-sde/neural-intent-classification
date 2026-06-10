# Neural Intent Classification

This repository contains a PyTorch-based intent classifier for the CLINC150
dataset. The current default model uses mean pooling over token embeddings, but
the project is structured so encoder backends are swappable. An LSTM backend is
already wired in through the same classifier interface. Tokenization is now
backed by a separate pluggable tokenizer layer so future text encoders such as
BPE can fit into the same training and inference flow.

## Repository Layout

```text
neural_intent_classification/
  data/         Dataset loading, preprocessing, tokenizer backends, dataset wrappers
  models/       Encoder backends and classifier head
  training/     Training loop and checkpointing
  utils/        Reproducibility helpers
docs/
  architecture.md
  development.md
AGENTS.md       Agent-facing repo context and conventions
train.py        Training entrypoint
predict.py      Inference entrypoint
explore.py      Dataset exploration entrypoint
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Mean pooling backend:

```bash
python3 train.py
```

Explicit tokenizer backend:

```bash
python3 train.py --tokenizer-type nltk_word
```

Train with the Hugging Face BPE tokenizer backend:

```bash
python3 train.py --tokenizer-type hf_bpe --tokenizer-vocab-size 2000
```

LSTM backend:

```bash
python3 train.py --encoder-type lstm
```

## Predict

```bash
python3 predict.py --text "set an alarm for tomorrow"
```

Optionally point to a specific checkpoint:

```bash
python3 predict.py --checkpoint checkpoints/best_model.pt --text "book me a cab"
```

You can also pass the tokenizer type explicitly when you need it to match the
checkpoint contract:

```bash
python3 predict.py --checkpoint checkpoints/best_model.pt --tokenizer-type hf_bpe --text "book me a cab"
```

## Explore the Dataset

```bash
python3 explore.py
```

## Notes

- Checkpoints are written to `checkpoints/` and are intentionally gitignored.
- `mean_pool` and `lstm` use the same training pipeline and checkpoint format.
- Tokenizer state is saved in checkpoints so inference restores the same text
  preprocessing path used during training.
- `hf_bpe` trains from the CLINC training split at runtime and stores the
  learned tokenizer state inside the checkpoint.
- Architecture and agent conventions are documented in [docs/architecture.md](docs/architecture.md)
  and [AGENTS.md](AGENTS.md).
