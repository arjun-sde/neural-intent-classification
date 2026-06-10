from __future__ import annotations

import argparse

import torch

from neural_intent_classification.config import ExperimentConfig
from neural_intent_classification.data.loader import load_clinc150, load_intent_mapping
from neural_intent_classification.data.preprocessing import normalize_label, pad_sequence
from neural_intent_classification.data.tokenizers import (
    BaseTokenizer,
    get_supported_tokenizer_types,
    load_tokenizer_from_checkpoint,
)
from neural_intent_classification.models.classifier import IntentClassifier


def load_checkpoint_artifacts(
    checkpoint_path: str,
    tokenizer_type_override: str | None = None,
) -> tuple[IntentClassifier, BaseTokenizer, ExperimentConfig]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    config = ExperimentConfig.from_dict(checkpoint.get("config"))
    if tokenizer_type_override is not None:
        config.tokenizer.tokenizer_type = tokenizer_type_override
    tokenizer = load_tokenizer_from_checkpoint(
        checkpoint=checkpoint,
        dataset_config=config.dataset,
        tokenizer_config=config.tokenizer,
    )
    model = IntentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=config.dataset.num_classes,
        padding_idx=config.dataset.pad_token_id,
        config=config.model,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def predict_text(
    text: str,
    model: IntentClassifier,
    tokenizer: BaseTokenizer,
    config: ExperimentConfig,
) -> int:
    token_ids = tokenizer.encode(text)
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
    model, tokenizer, config = load_checkpoint_artifacts(checkpoint_path)
    test_ds = load_clinc150(config.dataset)[2]
    correct = 0
    total = 0

    for row in test_ds:
        prediction = predict_text(
            text=row["utterance"],
            model=model,
            tokenizer=tokenizer,
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


def resolve_intent_name(
    label_id: int,
    intent_mapping,
    config: ExperimentConfig,
) -> str:
    if label_id == config.dataset.oos_label:
        return "oos"

    intents = intent_mapping["intents"]
    if label_id < 0 or label_id >= len(intents):
        raise ValueError(
            f"Predicted label id {label_id} is outside the intent mapping."
        )

    return intents[label_id]["name"]


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
    parser.add_argument(
        "--tokenizer-type",
        choices=get_supported_tokenizer_types(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, config = load_checkpoint_artifacts(
        args.checkpoint,
        tokenizer_type_override=args.tokenizer_type,
    )
    prediction = predict_text(
        text=args.text,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    intent_mapping = load_intent_mapping(config.dataset)
    intent_name = resolve_intent_name(
        prediction,
        intent_mapping,
        config,
    )

    print(f"Intent classified: {intent_name}")
    print(f"Predicted label id: {prediction}")

    if args.show_accuracy:
        accuracy = evaluate_test_set(args.checkpoint)
        print(f"Accuracy: {accuracy:.4f}")
