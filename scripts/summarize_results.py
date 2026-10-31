"""Combine per-run evaluation and training JSON files into a report-ready CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results_summary.csv"))
    args = parser.parse_args()

    evaluations = sorted((args.outputs_dir / "evaluation").rglob("*_metrics.json"))
    if not evaluations:
        raise SystemExit(f"No evaluation metric JSON files found under {args.outputs_dir / 'evaluation'}")

    fields = [
        "model", "architecture", "temperature", "test_examples",
        "bleu", "chrf", "inference_seconds", "seconds_per_example",
        "training_seconds", "parameter_count", "device", "batch_size",
        "max_input_length", "max_new_tokens", "sacrebleu_version",
    ]
    rows = []
    for path in evaluations:
        evaluation = read_json(path)
        model_id = str(evaluation.get("model", ""))
        model_name = model_id.rstrip("/").split("/")[-1]
        candidates = [
            args.outputs_dir / model_name / "training_metrics.json",
            args.outputs_dir / (model_name.lower()) / "training_metrics.json",
        ]
        training = next((read_json(p) for p in candidates if p.is_file()), {})
        row = {
            "model": model_id,
            "architecture": evaluation.get("architecture", ""),
            "temperature": evaluation.get("temperature", ""),
            "test_examples": evaluation.get("test_examples", ""),
            "bleu": evaluation.get("bleu", ""),
            "chrf": evaluation.get("chrf", ""),
            "inference_seconds": evaluation.get("inference_seconds", ""),
            "seconds_per_example": evaluation.get("seconds_per_example", ""),
            "training_seconds": training.get("training_seconds_wall_clock", "N/A if prompted"),
            "parameter_count": evaluation.get("parameter_count", training.get("parameter_count", "")),
            "device": evaluation.get("device", ""),
            "batch_size": evaluation.get("batch_size", ""),
            "max_input_length": evaluation.get("max_input_length", ""),
            "max_new_tokens": evaluation.get("max_new_tokens", ""),
            "sacrebleu_version": evaluation.get("sacrebleu_version", ""),
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} run rows to {args.output}")


if __name__ == "__main__":
    main()
