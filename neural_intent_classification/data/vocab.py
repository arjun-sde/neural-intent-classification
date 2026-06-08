from __future__ import annotations

from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

from neural_intent_classification.config import DatasetConfig


def download_nltk_resources() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [
        token
        for token in tokens
        if any(char.isalnum() for char in token)
    ]


def build_vocab(
    train_dataset,
    config: DatasetConfig,
) -> dict[str, int]:
    word_freq: Counter[str] = Counter()

    for row in train_dataset:
        word_freq.update(tokenize(row["utterance"]))

    vocab = {
        config.pad_token: config.pad_token_id,
        config.unk_token: config.unk_token_id,
    }

    for word in word_freq:
        vocab[word] = len(vocab)

    return vocab


def text_to_ids(
    text: str,
    vocab: dict[str, int],
    config: DatasetConfig,
) -> list[int]:
    token_ids = [
        vocab.get(token, config.unk_token_id)
        for token in tokenize(text)
    ]
    return token_ids or [config.unk_token_id]
