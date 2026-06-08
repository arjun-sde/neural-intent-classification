from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_intent_classification.config import ExperimentConfig
from neural_intent_classification.data.dataset import IntentDataset
from neural_intent_classification.data.loader import load_clinc150
from neural_intent_classification.data.vocab import (
    build_vocab,
    download_nltk_resources,
)
from neural_intent_classification.models.classifier import IntentClassifier
from neural_intent_classification.utils.reproducibility import (
    seed_worker,
    set_seed,
)


def build_dataloaders(config: ExperimentConfig):
    train_ds, val_ds, test_ds = load_clinc150(config.dataset)
    vocab = build_vocab(train_ds, config.dataset)

    train_dataset = IntentDataset(
        dataset=train_ds,
        vocab=vocab,
        config=config.dataset,
    )
    val_dataset = IntentDataset(
        dataset=val_ds,
        vocab=vocab,
        config=config.dataset,
    )
    test_dataset = IntentDataset(
        dataset=test_ds,
        vocab=vocab,
        config=config.dataset,
    )

    generator = torch.Generator().manual_seed(config.training.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
    )

    return vocab, train_loader, val_loader, test_loader


def evaluate(
    model: IntentClassifier,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                lengths=lengths,
            )
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(data_loader), correct / total


def build_model_summary(
    model: IntentClassifier,
    config: ExperimentConfig,
    vocab_size: int,
) -> str:
    lines = [
        "",
        "=" * 80,
        "MODEL SUMMARY",
        "=" * 80,
        f"Vocabulary Size: {vocab_size:,}",
        f"Encoder Type: {config.model.encoder_type}",
        f"Embedding Dimension: {config.model.embedding_dim}",
        f"Classifier Hidden Dimension: {config.model.hidden_dim}",
        f"Number of Classes: {config.dataset.num_classes}",
        f"Max Sequence Length: {config.dataset.max_length}",
        f"Batch Size: {config.training.batch_size}",
        f"Seed: {config.training.seed}",
        "",
        "Parameter Breakdown",
    ]

    for name, param in model.named_parameters():
        lines.append(
            f"{name:<35}"
            f"Shape={str(tuple(param.shape)):<20}"
            f"Params={param.numel():,}"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    lines.extend(
        [
            "",
            f"Trainable Parameters: {trainable_params:,}",
            f"Total Parameters: {total_params:,}",
            "=" * 80,
        ]
    )
    return "\n".join(lines)


def build_checkpoint_payload(
    model: IntentClassifier,
    vocab: dict[str, int],
    config: ExperimentConfig,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": config.to_dict(),
    }


def train_model(config: ExperimentConfig) -> None:
    download_nltk_resources()
    set_seed(config.training.seed)

    os.makedirs(
        config.training.checkpoint_dir,
        exist_ok=True,
    )

    vocab, train_loader, val_loader, test_loader = build_dataloaders(config)

    device = torch.device(config.training.device)
    model = IntentClassifier(
        vocab_size=len(vocab),
        num_classes=config.dataset.num_classes,
        padding_idx=config.dataset.pad_token_id,
        config=config.model,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
    )

    print(build_model_summary(model, config, len(vocab)))

    best_acc = 0.0
    best_checkpoint_path = os.path.join(
        config.training.checkpoint_dir,
        config.training.best_checkpoint_name,
    )
    final_checkpoint_path = os.path.join(
        config.training.checkpoint_dir,
        config.training.final_checkpoint_name,
    )

    for epoch in range(config.training.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                lengths=lengths,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                build_checkpoint_payload(model, vocab, config),
                best_checkpoint_path,
            )
            print(f"New best model saved (val acc: {val_acc:.4f})")

        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")

    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    torch.save(
        build_checkpoint_payload(model, vocab, config),
        final_checkpoint_path,
    )
    print("\nTraining complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-type",
        default="mean_pool",
        choices=["mean_pool", "lstm"],
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--unidirectional-lstm",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.encoder_type = args.encoder_type
    config.model.embedding_dim = args.embedding_dim
    config.model.hidden_dim = args.hidden_dim
    config.model.dropout = args.dropout
    config.model.lstm_hidden_dim = args.lstm_hidden_dim
    config.model.lstm_num_layers = args.lstm_num_layers
    config.model.lstm_bidirectional = not args.unidirectional_lstm
    config.dataset.max_length = args.max_length
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.num_epochs = args.num_epochs
    config.training.device = args.device
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    train_model(config)
