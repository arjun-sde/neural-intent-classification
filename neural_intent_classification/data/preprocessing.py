from __future__ import annotations

from neural_intent_classification.config import DatasetConfig


def normalize_label(
    label: int | None,
    config: DatasetConfig,
) -> int:
    if label is None:
        return config.oos_label
    return label


def pad_sequence(
    token_ids: list[int],
    max_length: int,
    pad_token_id: int,
) -> tuple[list[int], int]:
    clipped_tokens = token_ids[:max_length]
    length = len(clipped_tokens)
    padding_needed = max_length - length
    padded_tokens = clipped_tokens + [pad_token_id] * padding_needed
    return padded_tokens, length
