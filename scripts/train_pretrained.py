"""Fine-tune an encoder-decoder or decoder-only HF model on prepared CSV splits.

Examples:
  python scripts/train_pretrained.py --architecture seq2seq --model google/mt5-small
  python scripts/train_pretrained.py --architecture causal --model google/gemma-3-1b-it
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["seq2seq", "causal"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=4403)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    valid_path = args.data_dir / "validation.csv"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Run scripts/prepare_data.py before training.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.architecture == "causal" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cls = AutoModelForSeq2SeqLM if args.architecture == "seq2seq" else AutoModelForCausalLM
    model = model_cls.from_pretrained(args.model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data = load_dataset(
        "csv",
        data_files={"train": str(train_path), "validation": str(valid_path)},
    )

    def encode(batch):
        if args.architecture == "seq2seq":
            inputs = tokenizer(
                ["translate English to Hindi: " + text for text in batch["source"]],
                max_length=args.max_length,
                truncation=True,
            )
            targets = tokenizer(
                text_target=batch["target"],
                max_length=args.max_length,
                truncation=True,
            )
            inputs["labels"] = targets["input_ids"]
            return inputs

        prompts = [
            "Translate English to Hindi.\nEnglish: " + source + "\nHindi:"
            for source in batch["source"]
        ]
        encoded_prompts = tokenizer(prompts, add_special_tokens=False)
        full_text = [p + " " + t + tokenizer.eos_token for p, t in zip(prompts, batch["target"])]
        encoded_full = tokenizer(full_text, max_length=args.max_length, truncation=True)
        labels = []
        for prompt_ids, input_ids in zip(encoded_prompts["input_ids"], encoded_full["input_ids"]):
            prefix_length = min(len(prompt_ids), len(input_ids))
            labels.append([-100] * prefix_length + input_ids[prefix_length:])
        encoded_full["labels"] = labels
        return encoded_full

    tokenized = data.map(
        encode,
        batched=True,
        remove_columns=data["train"].column_names,
        desc="Tokenizing parallel text",
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    result = trainer.train()
    training_seconds = time.perf_counter() - started
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    training_metrics = dict(result.metrics)
    training_metrics["training_seconds_wall_clock"] = training_seconds
    training_metrics["parameter_count"] = sum(p.numel() for p in model.parameters())
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved fine-tuned model, tokenizer, and training metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
