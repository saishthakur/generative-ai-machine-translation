"""Train a randomly initialized encoder-decoder Transformer on prepared OPUS-100 CSVs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/basic-transformer"))
    parser.add_argument("--tokenizer", default="google/byt5-small")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=4403)
    parser.add_argument("--auto-prepare", action="store_true", help="prepare missing CSV splits before training")
    parser.add_argument("--prepare-train-rows", type=int, default=1000)
    parser.add_argument("--prepare-validation-rows", type=int, default=100)
    parser.add_argument("--prepare-test-rows", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        d_kv=args.d_model // args.heads,
        d_ff=args.d_model * 4,
        num_layers=args.layers,
        num_decoder_layers=args.layers,
        num_heads=args.heads,
        dropout_rate=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    model = T5ForConditionalGeneration(config)  # random initialization: no pretrained weights

    train_path = args.data_dir / "train.csv"
    valid_path = args.data_dir / "validation.csv"
    if args.auto_prepare and (not train_path.exists() or not valid_path.exists()):
        prepare_script = Path(__file__).with_name("prepare_data.py")
        subprocess.run(
            [
                sys.executable, str(prepare_script),
                "--output-dir", str(args.data_dir),
                "--train-rows", str(args.prepare_train_rows),
                "--validation-rows", str(args.prepare_validation_rows),
                "--test-rows", str(args.prepare_test_rows),
                "--seed", str(args.seed),
            ],
            check=True,
        )
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {valid_path}. Run scripts/prepare_data.py first "
            f"or pass --auto-prepare. Current working directory: {Path.cwd()}"
        )

    data = load_dataset(
        "csv",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        },
    )

    def encode(batch):
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

    tokenized = data.map(
        encode,
        batched=True,
        remove_columns=data["train"].column_names,
        desc="Tokenizing parallel text",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
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
    training_metrics["architecture"] = "randomly_initialized_encoder_decoder"
    training_metrics["tokenizer"] = args.tokenizer
    training_metrics["max_length"] = args.max_length
    training_metrics["batch_size"] = args.batch_size
    training_metrics["epochs_requested"] = args.epochs
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved randomly initialized Transformer and training metrics to {args.output_dir}")

