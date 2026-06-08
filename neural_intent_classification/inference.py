from __future__ import annotations

import argparse

import torch

from neural_intent_classification.config import ExperimentConfig
from neural_intent_classification.data.loader import load_clinc150, load_intent_mapping
from neural_intent_classification.data.preprocessing import normalize_label, pad_sequence
from neural_intent_classification.data.vocab import (
    download_nltk_resources,
    text_to_ids,
)
from neural_intent_classification.models.classifier import IntentClassifier


def load_checkpoint_artifacts(
    checkpoint_path: str,
) -> tuple[IntentClassifier, dict[str, int], ExperimentConfig]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    config = ExperimentConfig.from_dict(checkpoint.get("config"))
    vocab = checkpoint["vocab"]
    model = IntentClassifier(
        vocab_size=len(vocab),
        num_classes=config.dataset.num_classes,
        padding_idx=config.dataset.pad_token_id,
        config=config.model,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, config


def predict_text(
    text: str,
    model: IntentClassifier,
    vocab: dict[str, int],
    config: ExperimentConfig,
) -> int:
    token_ids = text_to_ids(
        text=text,
        vocab=vocab,
        config=config.dataset,
    )
    padded_ids, length = pad_sequence(
        token_ids=token_ids,
        max_length=config.dataset.max_length,
        pad_token_id=config.dataset.pad_token_id,
    )
    input_tensor = torch.tensor([padded_ids], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        logits = model(
            input_ids=input_tensor,
            lengths=lengths,
        )
        return logits.argmax(dim=1).item()


def evaluate_test_set(
    checkpoint_path: str,
) -> float:
    download_nltk_resources()
    model, vocab, config = load_checkpoint_artifacts(checkpoint_path)
    test_ds = load_clinc150(config.dataset)[2]
    correct = 0
    total = 0

    for row in test_ds:
        prediction = predict_text(
            text=row["utterance"],
            model=model,
            vocab=vocab,
            config=config,
        )
        label = normalize_label(
            row["label"],
            config.dataset,
        )
        if prediction == label:
            correct += 1
        total += 1

    return correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--text",
        required=True,
    )
    parser.add_argument(
        "--show-accuracy",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_nltk_resources()
    model, vocab, config = load_checkpoint_artifacts(args.checkpoint)
    prediction = predict_text(
        text=args.text,
        model=model,
        vocab=vocab,
        config=config,
    )
    intent_mapping = load_intent_mapping(config.dataset)
    intent_name = intent_mapping["intents"][prediction]["name"]

    print(f"Intent classified: {intent_name}")
    print(f"Predicted label id: {prediction}")

    if args.show_accuracy:
        accuracy = evaluate_test_set(args.checkpoint)
        print(f"Accuracy: {accuracy:.4f}")
