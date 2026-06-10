from __future__ import annotations

from neural_intent_classification.config import (
    DatasetConfig,
    TokenizerConfig,
)
from neural_intent_classification.data.tokenizers import build_tokenizer


def _default_word_tokenizer(
    config: DatasetConfig | None = None,
):
    dataset_config = config or DatasetConfig()
    tokenizer = build_tokenizer(
        dataset_config=dataset_config,
        tokenizer_config=TokenizerConfig(),
    )
    tokenizer.prepare()
    return tokenizer


def download_nltk_resources() -> None:
    _default_word_tokenizer()


def tokenize(text: str) -> list[str]:
    return _default_word_tokenizer().tokenize(text)


def build_vocab(
    train_dataset,
    config: DatasetConfig,
) -> dict[str, int]:
    tokenizer = _default_word_tokenizer(config)
    tokenizer.fit(
        row["utterance"]
        for row in train_dataset
    )
    return tokenizer.vocab or {}


def text_to_ids(
    text: str,
    vocab: dict[str, int],
    config: DatasetConfig,
) -> list[int]:
    tokenizer = _default_word_tokenizer(config)
    tokenizer.load_state_dict({"vocab": vocab})
    return tokenizer.encode(text)
