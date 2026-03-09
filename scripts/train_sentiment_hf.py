import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover - import guard for optional deps
    raise SystemExit(
        "Missing optional dependencies for the Hugging Face trainer. "
        "Install transformers and accelerate first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SST-2 classifier with Hugging Face Trainer."
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Pretrained model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sst2-hf",
        help="Directory used for checkpoints and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_subset(dataset, max_train_samples, max_eval_samples):
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(max_train_samples, len(dataset["train"])))
        )
    if max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )
    return dataset


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("glue", "sst2")
    dataset = maybe_subset(dataset, args.max_train_samples, args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(f"Train samples: {len(tokenized['train'])}")
    print(f"Validation samples: {len(tokenized['validation'])}")
    print(f"Output dir: {output_dir}")

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model()

    metrics = {
        "train": train_result.metrics,
        "eval": eval_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
