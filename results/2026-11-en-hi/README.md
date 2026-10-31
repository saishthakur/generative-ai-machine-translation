# Experiment artifact guide

This directory contains the submitted English-to-Hindi experiment outputs. It does not contain the source corpus or model checkpoints.

## Files

- `results_summary.csv` — one row per evaluated model and decoding-temperature run.
- `basic-transformer/training_metrics.json` and `mt5-small/training_metrics.json` — training runtime, configuration, and parameter counts for the two fine-tuned systems.
- `evaluation/<run>/` — the predictions CSV and the corresponding metric JSON for each run. Each CSV contains 1,000 rows with `source`, `reference`, and `prediction` columns.
- `figures/` — per-model result plots, a combined comparison, temperature plots, and the original chart exported by the submitted notebook.

## Interpretation notes

BLEU and chrF values use the 0–100 scale reported by SacreBLEU 2.6.0. Temperature 0 is deterministic decoding; positive temperatures use sampling. Sampled results can vary between runs. Run settings and interpretation are documented in [`docs/RESULTS_AND_ANALYSIS.md`](../../docs/RESULTS_AND_ANALYSIS.md).

Prediction outputs may contain errors or sensitive examples. Treat them as research artifacts, do not rely on them as professional translations, and review content before quoting or redistributing it.
