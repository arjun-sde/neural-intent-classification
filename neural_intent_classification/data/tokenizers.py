from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Iterable

import nltk
from nltk.tokenize import word_tokenize
from tokenizers import Tokenizer
from tokenizers import models as hf_models
from tokenizers import normalizers as hf_normalizers
from tokenizers import pre_tokenizers as hf_pre_tokenizers
from tokenizers import trainers as hf_trainers

from neural_intent_classification.config import (
    DatasetConfig,
    TokenizerConfig,
)


class BaseTokenizer(ABC):
    tokenizer_type: str

    def __init__(
        self,
        dataset_config: DatasetConfig,
        config: TokenizerConfig,
    ) -> None:
        self.dataset_config = dataset_config
        self.config = config

    def prepare(self) -> None:
        """Fetch resources required by the tokenizer backend."""

    @abstractmethod
    def fit(self, texts: Iterable[str]) -> None:
        """Learn tokenizer state from training text when required."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return token strings for inspection or statistics."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert raw text into model token ids."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Serialize tokenizer state for checkpoints."""

    @abstractmethod
    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore tokenizer state from a checkpoint payload."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the number of ids available to the embedding table."""

    @property
    def vocab(self) -> dict[str, int] | None:
        return None


class NltkWordTokenizer(BaseTokenizer):
    tokenizer_type = "nltk_word"

    def __init__(
        self,
        dataset_config: DatasetConfig,
        config: TokenizerConfig,
    ) -> None:
        super().__init__(dataset_config, config)
        self._vocab = {
            self.dataset_config.pad_token: self.dataset_config.pad_token_id,
            self.dataset_config.unk_token: self.dataset_config.unk_token_id,
        }

    def prepare(self) -> None:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    def fit(self, texts: Iterable[str]) -> None:
        word_freq: Counter[str] = Counter()

        for text in texts:
            word_freq.update(self.tokenize(text))

        self._vocab = {
            self.dataset_config.pad_token: self.dataset_config.pad_token_id,
            self.dataset_config.unk_token: self.dataset_config.unk_token_id,
        }

        for token in word_freq:
            self._vocab[token] = len(self._vocab)

    def tokenize(self, text: str) -> list[str]:
        raw_tokens = word_tokenize(text)
        tokens = [
            token.lower()
            if self.config.lowercase
            else token
            for token in raw_tokens
        ]

        if not self.config.filter_non_alnum_tokens:
            return tokens

        return [
            token
            for token in tokens
            if any(char.isalnum() for char in token)
        ]

    def encode(self, text: str) -> list[int]:
        token_ids = [
            self._vocab.get(token, self.dataset_config.unk_token_id)
            for token in self.tokenize(text)
        ]
        return token_ids or [self.dataset_config.unk_token_id]

    def state_dict(self) -> dict[str, Any]:
        return {"vocab": self._vocab}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        vocab = payload.get("vocab")
        if vocab is None:
            raise ValueError("Tokenizer state is missing 'vocab'.")

        self._vocab = dict(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab


class HuggingFaceBPETokenizer(BaseTokenizer):
    tokenizer_type = "hf_bpe"

    def __init__(
        self,
        dataset_config: DatasetConfig,
        config: TokenizerConfig,
    ) -> None:
        super().__init__(dataset_config, config)
        self._tokenizer = self._build_untrained_tokenizer()

    def _build_untrained_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(
            hf_models.BPE(
                unk_token=self.dataset_config.unk_token,
            )
        )

        normalizer_steps = [hf_normalizers.NFKC()]
        if self.config.lowercase:
            normalizer_steps.append(hf_normalizers.Lowercase())
        tokenizer.normalizer = hf_normalizers.Sequence(
            normalizer_steps
        )
        tokenizer.pre_tokenizer = hf_pre_tokenizers.Whitespace()
        return tokenizer

    def fit(self, texts: Iterable[str]) -> None:
        trainer = hf_trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=[
                self.dataset_config.pad_token,
                self.dataset_config.unk_token,
            ],
        )
        self._tokenizer = self._build_untrained_tokenizer()
        self._tokenizer.train_from_iterator(
            texts,
            trainer=trainer,
        )

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.encode(text).tokens

    def encode(self, text: str) -> list[int]:
        token_ids = self._tokenizer.encode(text).ids
        return token_ids or [self.dataset_config.unk_token_id]

    def state_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_json": self._tokenizer.to_str(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        tokenizer_json = payload.get("tokenizer_json")
        if tokenizer_json is not None:
            self._tokenizer = Tokenizer.from_str(tokenizer_json)
            return

        vocab = payload.get("vocab")
        if vocab is None:
            raise ValueError(
                "Tokenizer state is missing both tokenizer_json and vocab."
            )

        merges = payload.get("merges")
        if merges is None:
            raise ValueError(
                "Tokenizer state is missing 'merges' for BPE vocab restore."
            )

        self._tokenizer = Tokenizer(
            hf_models.BPE(
                vocab=vocab,
                merges=merges,
                unk_token=self.dataset_config.unk_token,
            )
        )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()


TOKENIZER_REGISTRY = {
    HuggingFaceBPETokenizer.tokenizer_type: HuggingFaceBPETokenizer,
    NltkWordTokenizer.tokenizer_type: NltkWordTokenizer,
}


def build_tokenizer(
    dataset_config: DatasetConfig,
    tokenizer_config: TokenizerConfig,
) -> BaseTokenizer:
    tokenizer_cls = TOKENIZER_REGISTRY.get(
        tokenizer_config.tokenizer_type
    )
    if tokenizer_cls is None:
        raise ValueError(
            "Unsupported tokenizer type: "
            f"{tokenizer_config.tokenizer_type}"
        )

    return tokenizer_cls(dataset_config, tokenizer_config)


def get_supported_tokenizer_types() -> list[str]:
    return sorted(TOKENIZER_REGISTRY)


def serialize_tokenizer_state(
    tokenizer: BaseTokenizer,
) -> dict[str, Any]:
    return {
        "tokenizer_type": tokenizer.tokenizer_type,
        "state": tokenizer.state_dict(),
    }


def load_tokenizer_from_checkpoint(
    checkpoint: dict[str, Any],
    dataset_config: DatasetConfig,
    tokenizer_config: TokenizerConfig,
) -> BaseTokenizer:
    tokenizer = build_tokenizer(
        dataset_config=dataset_config,
        tokenizer_config=tokenizer_config,
    )
    tokenizer.prepare()

    tokenizer_payload = checkpoint.get("tokenizer_state")
    if tokenizer_payload is None:
        legacy_vocab = checkpoint.get("vocab")
        if legacy_vocab is None:
            raise ValueError(
                "Checkpoint is missing tokenizer_state and vocab."
            )
        tokenizer.load_state_dict({"vocab": legacy_vocab})
        return tokenizer

    checkpoint_tokenizer_type = tokenizer_payload.get(
        "tokenizer_type",
        tokenizer_config.tokenizer_type,
    )
    if checkpoint_tokenizer_type != tokenizer.tokenizer_type:
        raise ValueError(
            "Checkpoint tokenizer type does not match config: "
            f"{checkpoint_tokenizer_type} != "
            f"{tokenizer.tokenizer_type}"
        )

    tokenizer.load_state_dict(tokenizer_payload["state"])
    return tokenizer
