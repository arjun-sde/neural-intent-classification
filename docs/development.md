# Development

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the default model:

```bash
python3 train.py
```

Train with LSTM backend:

```bash
python3 train.py --encoder-type lstm
```

Run a prediction:

```bash
python3 predict.py --text "set an alarm for tomorrow"
```

Inspect dataset statistics:

```bash
python3 explore.py
```

## Backend Workflow

When changing model architecture, keep the following stable:

1. `IntentDataset` should continue returning generic sequence data.
2. `IntentClassifier` should remain the single public model entrypoint.
3. Checkpoints must store the config needed for full restoration.

## Practical Notes

- The dataset comes from Hugging Face: `DeepPavlov/clinc150`.
- NLTK tokenization resources are downloaded at runtime if missing.
- The training script writes `best_model.pt` and `final_model.pt` to
  `checkpoints/`.
