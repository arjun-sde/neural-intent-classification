from __future__ import annotations

import torch
import torch.nn as nn

from neural_intent_classification.config import ModelConfig


class MeanPoolingEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        padding_idx: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.output_dim = embedding_dim

    def forward(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        del lengths
        mask = (input_ids != self.padding_idx).unsqueeze(-1)
        masked_embeddings = embeddings * mask
        token_counts = mask.sum(dim=1).clamp(min=1).to(embeddings.dtype)
        return masked_embeddings.sum(dim=1) / token_counts


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
        )
        self.output_dim = config.lstm_hidden_dim * (
            2 if config.lstm_bidirectional else 1
        )
        self.bidirectional = config.lstm_bidirectional

    def forward(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_state, _) = self.lstm(packed)

        if self.bidirectional:
            return torch.cat(
                (hidden_state[-2], hidden_state[-1]),
                dim=1,
            )

        return hidden_state[-1]


def build_encoder(
    embedding_dim: int,
    padding_idx: int,
    config: ModelConfig,
) -> nn.Module:
    if config.encoder_type == "mean_pool":
        return MeanPoolingEncoder(
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    if config.encoder_type == "lstm":
        return LSTMEncoder(
            embedding_dim=embedding_dim,
            config=config,
        )

    raise ValueError(
        f"Unsupported encoder type: {config.encoder_type}"
    )
