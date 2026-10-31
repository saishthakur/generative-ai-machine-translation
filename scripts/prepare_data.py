"""Download a reproducible, bounded OPUS-100 English-Hindi subset.

The dataset already provides train/validation/test splits. This script preserves those
splits, shuffles only the training stream with a fixed seed, and writes CSV files
under data/processed. It does not upload or commit dataset rows.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def write_split(rows, path: Path, source_lang: str, target_lang: str, limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        writer.writeheader()
        for row in rows:
            pair = row["translation"]
            source = (pair.get(source_lang) or "").strip()
            target = (pair.get(target_lang) or "").strip()
            if not source or not target:
                continue
            writer.writerow({"source": source, "target": target})
            count += 1
            if count >= limit:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Helsinki-NLP/opus-100")
    parser.add_argument("--config", default="en-hi", help="OPUS-100 pair, e.g. en-hi")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="hi")
    parser.add_argument("--train-rows", type=int, default=20000)
    parser.add_argument("--validation-rows", type=int, default=1000)
    parser.add_argument("--test-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4403)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    for split, limit in (
        ("train", args.train_rows),
        ("validation", args.validation_rows),
        ("test", args.test_rows),
    ):
        stream = load_dataset(
            args.dataset, args.config, split=split, streaming=True
        )
        if split == "train":
            stream = stream.shuffle(seed=args.seed, buffer_size=10_000)
        count = write_split(
            stream,
            args.output_dir / f"{split}.csv",
            args.source_lang,
            args.target_lang,
            limit,
        )
        print(f"{split}: wrote {count} sentence pairs to {args.output_dir}")

    print(f"Dataset: {args.dataset} ({args.config}); seed={args.seed}")
    print("Keep data/processed local unless the dataset terms explicitly allow redistribution.")


if __name__ == "__main__":
    main()
