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

Train with an explicit tokenizer backend:

```bash
python3 train.py --tokenizer-type nltk_word
```

Train with the BPE tokenizer backend:

```bash
python3 train.py --tokenizer-type hf_bpe --tokenizer-vocab-size 2000 --tokenizer-min-frequency 2
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
3. Tokenizer backends should be selected and restored through
   `data/tokenizers.py`.
4. Checkpoints must store the config and tokenizer state needed for full
   restoration.

## Practical Notes

- The dataset comes from Hugging Face: `DeepPavlov/clinc150`.
- Tokenizer resources are prepared at runtime by the active tokenizer backend.
- The default tokenizer backend is `nltk_word`, which uses NLTK word
  tokenization plus normalization before id lookup.
- The `hf_bpe` tokenizer backend uses the `tokenizers` library and is trained
  on the CLINC training split during dataloader construction.
- The training script writes `best_model.pt` and `final_model.pt` to
  `checkpoints/`.
