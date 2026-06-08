from __future__ import annotations

from datasets import load_dataset

from neural_intent_classification.config import DatasetConfig


def load_clinc150(config: DatasetConfig):
    dataset = load_dataset(config.dataset_name)
    return dataset["train"], dataset["validation"], dataset["test"]


def load_intent_mapping(config: DatasetConfig):
    return load_dataset(
        config.dataset_name,
        "intents",
    )
