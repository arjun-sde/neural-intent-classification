from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetConfig:
    dataset_name: str = "DeepPavlov/clinc150"
    max_length: int = 24
    num_classes: int = 151
    oos_label: int = 150
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    pad_token_id: int = 0
    unk_token_id: int = 1


@dataclass(slots=True)
class TokenizerConfig:
    tokenizer_type: str = "nltk_word"
    lowercase: bool = True
    filter_non_alnum_tokens: bool = True
    vocab_size: int = 2000
    min_frequency: int = 2


@dataclass(slots=True)
class ModelConfig:
    encoder_type: str = "mean_pool"
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.3
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 1
    lstm_bidirectional: bool = True


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 20
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    best_checkpoint_name: str = "best_model.pt"
    final_checkpoint_name: str = "final_model.pt"


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
    ) -> "ExperimentConfig":
        if not payload:
            return cls()

        dataset = DatasetConfig(**payload.get("dataset", {}))
        tokenizer = TokenizerConfig(**payload.get("tokenizer", {}))
        model = ModelConfig(**payload.get("model", {}))
        training = TrainingConfig(**payload.get("training", {}))
        return cls(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            training=training,
        )
