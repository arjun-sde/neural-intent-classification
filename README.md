# Neural Intent Classification

This repository contains a PyTorch-based intent classifier for the CLINC150
dataset. The current default model uses mean pooling over token embeddings, but
the project is structured so encoder backends are swappable. An LSTM backend is
already wired in through the same classifier interface.

## Repository Layout

```text
neural_intent_classification/
  data/         Dataset loading, preprocessing, vocabulary, dataset wrappers
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

## Explore the Dataset

```bash
python3 explore.py
```

## Notes

- Checkpoints are written to `checkpoints/` and are intentionally gitignored.
- `mean_pool` and `lstm` use the same training pipeline and checkpoint format.
- Architecture and agent conventions are documented in [docs/architecture.md](docs/architecture.md)
  and [AGENTS.md](AGENTS.md).
