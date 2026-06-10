from __future__ import annotations

import numpy as np

from neural_intent_classification.config import ExperimentConfig
from neural_intent_classification.data.loader import load_clinc150
from neural_intent_classification.data.preprocessing import normalize_label
from neural_intent_classification.data.tokenizers import (
    build_tokenizer,
)


def main() -> None:
    config = ExperimentConfig()
    tokenizer = build_tokenizer(
        dataset_config=config.dataset,
        tokenizer_config=config.tokenizer,
    )
    tokenizer.prepare()

    train_ds, _, _ = load_clinc150(config.dataset)
    lengths = [
        len(tokenizer.tokenize(row["utterance"]))
        for row in train_ds
    ]

    print("=" * 50)
    print(f"Train Samples: {len(train_ds)}")
    print(f"Min Length: {min(lengths)}")
    print(f"Max Length: {max(lengths)}")
    print(f"Average Length: {np.mean(lengths):.2f}")
    print(f"95th Percentile: {np.percentile(lengths, 95)}")
    print(f"99th Percentile: {np.percentile(lengths, 99)}")
    print("=" * 50)

    tokenizer.fit(
        row["utterance"]
        for row in train_ds
    )
    labels = {
        normalize_label(row["label"], config.dataset)
        for row in train_ds
    }

    print(f"Tokenizer Type: {config.tokenizer.tokenizer_type}")
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Unique Labels: {len(labels)}")
    print(labels)
