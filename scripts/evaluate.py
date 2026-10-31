"""Generate translations and calculate corpus BLEU/chrF and inference time."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["seq2seq", "causal"], required=True)
    parser.add_argument("--model", required=True, help="Local fine-tuned path or Hub model ID")
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.architecture == "causal":
        tokenizer.padding_side = "left"
    cls = AutoModelForSeq2SeqLM if args.architecture == "seq2seq" else AutoModelForCausalLM
    model = cls.from_pretrained(args.model).to(device).eval()

    rows = list(csv.DictReader(args.test.open(encoding="utf-8")))
    sources = [r["source"] for r in rows]
    references = [r["target"] for r in rows]
    predictions = []
    started = time.perf_counter()

    for start in range(0, len(rows), args.batch_size):
        batch_sources = sources[start : start + args.batch_size]
        if args.architecture == "seq2seq":
            prompts = ["translate English to Hindi: " + s for s in batch_sources]
        else:
            prompts = []
            for source in batch_sources:
                user_text = "Translate English to Hindi.\nEnglish: " + source + "\nHindi:"
                if getattr(tokenizer, "chat_template", None):
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    prompt = user_text
                prompts.append(prompt)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            add_special_tokens=args.architecture != "causal",
        ).to(device)
        generation = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": 1,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.temperature > 0:
            generation["temperature"] = args.temperature
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
        if args.architecture == "causal":
            prompt_width = inputs["input_ids"].shape[1]
            output = output[:, prompt_width:]
        predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / (args.model.rstrip("/").split("/")[-1] + "_predictions.csv")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(
            {"source": s, "reference": r, "prediction": p}
            for s, r, p in zip(sources, references, predictions)
        )

    params = sum(p.numel() for p in model.parameters())
    metrics = {
        "model": args.model,
        "architecture": args.architecture,
        "test_examples": len(rows),
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf": sacrebleu.corpus_chrf(predictions, [references]).score,
        "inference_seconds": elapsed,
        "seconds_per_example": elapsed / max(1, len(rows)),
        "parameter_count": params,
        "temperature": args.temperature,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "sacrebleu_version": sacrebleu.__version__,
    }
    output_json = args.output_dir / (output_csv.stem + "_metrics.json")
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
