# Experiment results and analysis

## Run configuration

- Language pair: English → Hindi (`en-hi`), OPUS-100.
- Subset described by the submitted notebook: 20,000 training, 1,000 validation, and 1,000 test examples; seed 4403.
- Test set: 1,000 examples per run.
- Metrics: SacreBLEU 2.6.0 corpus BLEU and chrF (0–100 scales).
- Device: CUDA GPU, as recorded in the metrics files.
- Basic Transformer: 3 epochs, batch size 16 in training; evaluation batch size 32.
- mT5-small: 3 epochs, batch size 8 in training; evaluation batch size 32.
- LLaMA 3.2 1B Instruct and Gemma 3 1B IT: prompted, no fine-tuning; evaluation batch size 8.
- Encoder-decoder generation used up to 256 new tokens; decoder-only generation used up to 128. Runtime comparisons are not fully controlled because batch size and generation limits differ.

## Primary comparison (temperature 0)

| Model | Architecture | BLEU | chrF | Training / fine-tuning | Inference total | Inference per example | Parameters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Basic Transformer | Encoder-decoder | 0.304 | 8.121 | 434.84 s | 60.54 s | 0.0605 s | 7,411,456 |
| mT5-small | Encoder-decoder | **4.797** | 19.206 | 2,811.66 s | 136.26 s | 0.1363 s | 300,176,768 |
| LLaMA 3.2 1B Instruct | Decoder-only | 4.032 | **22.524** | Prompted; not fine-tuned | 1,367.78 s | 1.3678 s | 1,235,814,400 |
| Gemma 3 1B IT | Decoder-only | 1.086 | 13.483 | Prompted; not fine-tuned | 1,524.58 s | 1.5246 s | 999,885,952 |

These are the measurements from the submitted result files. mT5 leads on BLEU; LLaMA leads on chrF. A single overall winner cannot be established from these two metrics alone. Scores are low, especially for the scratch baseline, so the system outputs need substantial improvement and human review.

## Temperature results for prompted models

| Model | Temperature | BLEU | chrF | Inference seconds |
| --- | ---: | ---: | ---: | ---: |
| LLaMA 3.2 1B Instruct | 0.0 | 4.032 | 22.524 | 1,367.78 |
| LLaMA 3.2 1B Instruct | 0.3 | 3.956 | 22.490 | 1,421.19 |
| LLaMA 3.2 1B Instruct | 0.7 | 3.170 | 21.272 | 1,429.32 |
| Gemma 3 1B IT | 0.0 | 1.086 | 13.483 | 1,524.58 |
| Gemma 3 1B IT | 0.3 | 1.068 | 13.488 | 1,613.18 |
| Gemma 3 1B IT | 0.7 | 1.064 | 13.337 | 1,614.43 |

In these runs, increasing temperature did not improve aggregate metric scores. LLaMA's largest drop is at 0.7. Gemma's chrF is effectively unchanged at 0.3 and falls slightly at 0.7. Since each temperature was run once, small score changes should not be interpreted as statistically reliable.

## Translation inspection and responsible-use observations

One example from the stored test predictions illustrates why aggregate scores are insufficient:

> **Source:** Give shots of injections or pills, but he must be alright soon.
>
> **Reference:** सुई लगाओ या गोली खिलाओ लेकिन इसे जल्दी से ठीक करो.
>
> **LLaMA output:** English: दिल्ली में एक चिकित्सक के पास जाने के बाद, उन्हें जल्द ही ठीक होने की जरूरत है।

The LLaMA output introduces a Delhi doctor that is not in the source, while changing the imperative and treatment details. This is a potentially harmful hallucination in a medical context. In the same row, the scratch Transformer's output is largely repetitive and mT5's output is not a faithful rendering. Gemma returned multiple alternative translations with explanatory text rather than one clean translation, which can break downstream use and obscure the intended meaning. This small sample is illustrative, not a systematic human evaluation.

Potential risks include omission or addition of meaning, hallucinated entities, wrong medical or other domain terminology, gender/register choices, regional and cultural bias in the parallel corpus, and inappropriate handling of sensitive content. Mitigations include human review for consequential translations, testing with domain-specific and regional examples, terminology controls, privacy safeguards, disclosure of machine translation, and abstention/escalation when an output cannot be validated. No fluent-speaker evaluation, bias audit, or confidence calibration is included in this experiment.

## Figures and raw artifacts

The `figures/` directory contains model-specific metric charts, the combined comparison, and temperature charts. `results_summary.csv` is the consolidated source of plotted measurements. For each model/temperature run, `evaluation/` contains the exact predictions CSV and its metrics JSON. Training metrics are stored for the two fine-tuned systems. See [README](../README.md) for setup and reproduction notes.
