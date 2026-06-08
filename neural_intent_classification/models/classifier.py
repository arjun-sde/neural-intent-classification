from __future__ import annotations

import torch
import torch.nn as nn

from neural_intent_classification.config import ModelConfig
from neural_intent_classification.models.encoders import build_encoder


class IntentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        padding_idx: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=padding_idx,
        )
        self.encoder = build_encoder(
            embedding_dim=config.embedding_dim,
            padding_idx=padding_idx,
            config=config,
        )
        self.fc1 = nn.Linear(
            self.encoder.output_dim,
            config.hidden_dim,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(
            config.hidden_dim,
            num_classes,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        sentence_embedding = self.encoder(
            embeddings=embeddings,
            input_ids=input_ids,
            lengths=lengths,
        )
        hidden = self.fc1(sentence_embedding)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        return self.fc2(hidden)
