from __future__ import annotations

import torch
from torch.utils.data import Dataset

from neural_intent_classification.config import DatasetConfig
from neural_intent_classification.data.preprocessing import (
    normalize_label,
    pad_sequence,
)
from neural_intent_classification.data.vocab import text_to_ids


class IntentDataset(Dataset):
    def __init__(
        self,
        dataset,
        vocab: dict[str, int],
        config: DatasetConfig,
    ):
        self.dataset = dataset
        self.vocab = vocab
        self.config = config

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.dataset[idx]
        token_ids = text_to_ids(
            row["utterance"],
            self.vocab,
            self.config,
        )
        padded_ids, length = pad_sequence(
            token_ids,
            self.config.max_length,
            self.config.pad_token_id,
        )
        label = normalize_label(
            row["label"],
            self.config,
        )

        return {
            "input_ids": torch.tensor(
                padded_ids,
                dtype=torch.long,
            ),
            "lengths": torch.tensor(
                length,
                dtype=torch.long,
            ),
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }
