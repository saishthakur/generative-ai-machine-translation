# Generative AI for Machine Translation

A comparative machine-translation study for the Foundations of Generative AI (ICT4403) Assignment 3. The project will compare encoder-decoder and decoder-only approaches on the same parallel-text test set.

> **Status:** Project scaffold. The repository currently contains no implementation, selected dataset, or experiment results. Sections marked “Pending” must be completed after the experiments are run.

## Objective

Build and evaluate four systems for one selected source-to-target language pair:

1. A basic Transformer with an encoder-decoder architecture, trained on the selected parallel corpus.
2. mT5, fine-tuned for translation.
3. LLaMA, adapted with a suitable fine-tuning method or evaluated with a documented prompting approach.
4. Gemma, adapted with a suitable fine-tuning method or evaluated with a documented prompting approach.

Use a manageable model size and dataset subset that fit the available compute. Generate translations for the same held-out test examples and compare quality, resource requirements, and responsible-use considerations.

## Dataset

**Language pair:** Pending selection  
**Dataset and version:** Pending selection  
**Source link:** Pending selection  
**License / usage terms:** Pending verification

The final dataset description should identify the languages, source, license, number of sentence pairs used, filtering and preprocessing steps, and any deduplication or length limits. Record the exact train, validation, and test split method and random seed. Keep the test set separate from training and model selection.

## Models

| Model | Architecture | Planned adaptation |
| --- | --- | --- |
| Basic Transformer | Encoder-decoder | Train from scratch on the training split |
| mT5 | Encoder-decoder | Fine-tune a suitable pretrained checkpoint |
| LLaMA | Decoder-only | Fine-tune or use a documented translation prompt |
| Gemma | Decoder-only | Fine-tune or use a documented translation prompt |

For each model, document the exact checkpoint, tokenizer, parameter count, adaptation method, key settings, and hardware. For prompted models, include the prompt template. For fine-tuned models, report training settings and time.

## Experimental method

1. Prepare and document the parallel corpus, then create train, validation, and test splits.
2. Train or adapt each model using the training data; use validation data for model selection.
3. Translate the same test examples with all four models under documented, comparable decoding settings.
4. Save predictions and calculate BLEU and chrF with the metric implementation and version recorded.
5. Measure training or fine-tuning time where applicable, inference time, and parameter count. State the timing setup, hardware, batch size, and whether preprocessing is included.
6. Repeat decoder-only inference at multiple sampling temperatures and discuss how temperature affects translation quality and consistency. Use deterministic decoding as a reference where supported.
7. Inspect examples from all systems and discuss errors, bias, hallucinated information, cultural misrepresentation, and sensitive content.

## Results

Results have not yet been collected. Replace the pending cells after running the experiments; do not report estimates as measured results.

| Model | BLEU | chrF | Training / fine-tuning time | Inference time | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic Transformer | Pending | Pending | Pending | Pending | Pending |
| mT5 | Pending | Pending | Pending | Pending | Pending |
| LLaMA | Pending | Pending | Pending | Pending | Pending |
| Gemma | Pending | Pending | Pending | Pending | Pending |

### Temperature analysis

Pending. Report the tested temperatures, decoding settings, and evaluation scores for each setting. Include representative examples and explain any trade-off between fluency, consistency, and faithfulness.

### Comparison and conclusion

Pending. Compare the encoder-decoder and decoder-only systems in light of their parameter counts, pretraining, tokenization, adaptation method, translation quality, and compute cost. Identify the best-performing model only after evaluating the recorded results, and state the criteria used.

## Responsible use

Translation output can reflect biases in training data, omit or add meaning, misrepresent cultural context, or produce inappropriate translations of sensitive material. Review outputs with fluent speakers where possible, especially for high-impact or sensitive content. Preserve source meaning, avoid treating generated translations as verified facts, document known limitations, and apply human review before consequential use.

## Repository contents

At present, this repository contains only this README, a license file, and a .gitignore. Implementation, environment/dependency files, dataset references, predictions, plots, and results are pending.

## Running the project

No runnable implementation or dependency list is present yet. Add the notebook or source files and a reproducible environment specification before documenting installation and execution commands here. Do not commit restricted or licensed dataset content unless its terms permit redistribution; provide the official dataset source and download/preparation instructions instead.

## Reproducibility checklist

- [ ] Select language pair and publicly available parallel dataset; record source and license.
- [ ] Add preprocessing and documented train/validation/test splits.
- [ ] Implement and document all four systems.
- [ ] Record checkpoints, tokenizer, parameter counts, settings, hardware, and random seeds.
- [ ] Save test translations and metric outputs.
- [ ] Measure runtime and conduct temperature analysis.
- [ ] Add comparison, conclusion, representative examples, and ethics discussion.
- [ ] Add dependencies and verified execution instructions.
