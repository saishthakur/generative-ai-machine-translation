# Generative AI for Machine Translation

A comparative English-to-Hindi machine-translation project for Foundations of Generative AI (ICT4403), Assignment 3. The workflow covers a randomly initialized encoder-decoder Transformer, fine-tuned mT5, and prompted or fine-tuned LLaMA and Gemma systems.

> **Status:** Training and evaluation scripts are now in the repository. No model has been trained and no experiment results have been collected yet. Results must come from running the commands below; the repository does not claim performance figures.

## Objective

Compare encoder-decoder and decoder-only systems on the same held-out English-to-Hindi test examples. Evaluate translation quality, training or fine-tuning time, inference time, parameter count, and responsible-use concerns.

## Dataset

- **Corpus:** [OPUS-100, English-Hindi configuration](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- **Dataset source:** [OPUS](https://opus.nlpl.eu/)
- **Configuration:** `en-hi`; fields are parallel English (`en`) and Hindi (`hi) strings in the `translation` column.
- **Published split sizes:** 534,319 training pairs and 2,000 pairs in each of validation and test (as listed in the dataset card).
- **Working subset:** The preparation script defaults to 20,000 training, 1,000 validation, and 1,000 test examples. It keeps the corpus-provided validation/test splits and shuffles only the training stream with seed 4403.
- **License:** The Hugging Face dataset card currently lists the license as unknown. Check the corpus and underlying source terms for the intended use, cite OPUS-100, and do not redistribute downloaded data unless permitted.

OPUS-100's corpus card describes its language coverage and its pre-separated train, development, and test samples. See the [dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus-100) and the [OPUS-100 paper](https://aclanthology.org/2020.acl-main.148/). The default subset limits are intended to keep experiments manageable; state the actual counts and any additional filtering in the final report.

## Models

| System | Architecture | Workflow |
| --- | --- | --- |
| Basic Transformer | Encoder-decoder | T5-style Transformer configuration initialized from scratch; ByT5 tokenizer only |
| mT5 | Encoder-decoder | Fine-tune `google/mt5-small` |
| LLaMA | Decoder-only | Prompt the base model for translation, or fine-tune using the provided causal training path |
| Gemma | Decoder-only | Prompt the base model for translation, or fine-tune using the provided causal training path |

For the model comparison, use small checkpoints that fit the available hardware and record exact model IDs. Example compact checkpoints are [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [Gemma 3 1B Instruct](https://huggingface.co/google/gemma-3-1b-it). Llama and Gemma access may require accepting their respective terms and authenticating with Hugging Face. The LLaMA and Gemma evaluation prompts are English-to-Hindi instructions; document any prompt changes.

## Setup

Python 3.10-3.12 is recommended. Model downloads and training require internet access. A CUDA GPU is strongly recommended for fine-tuning mT5, LLaMA, or Gemma. The basic Transformer can run on CPU, but training is likely to be slow.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Step-by-step

### 1. Prepare the dataset subset

```bash
python scripts/prepare_data.py --train-rows 20000 --validation-rows 1000 --test-rows 1000 --seed 4403
```

This downloads only the selected split streams and writes `data/processed/{train,validation,test}.csv`. The script does not commit downloaded data. Use the same generated CSV test file for each model.

### 2. Train the basic Transformer from scratch

```bash
python scripts/train_basic_transformer.py --epochs 5 --batch-size 16
```

It saves model files and `training_metrics.json` under `outputs/basic-transformer/`. The model weights are randomly initialized; the ByT5 tokenizer is used only for text encoding.

### 3. Fine-tune mT5

```bash
python scripts/train_pretrained.py --architecture seq2seq --model google/mt5-small --output-dir outputs/mt5-small --epochs 1 --batch-size 4
```

For limited GPU memory, reduce the batch size and use `--gradient-checkpointing`. Record the actual settings used.

### 4. Prompt LLaMA and Gemma

The assignment permits suitable prompting for the decoder-only systems. Start with zero-shot prompting on the same test set:

```bash
python scripts/evaluate.py --architecture causal --model meta-llama/Llama-3.2-1B --temperature 0 --output-dir outputs/llama
python scripts/evaluate.py --architecture causal --model google/gemma-3-1b-it --temperature 0 --output-dir outputs/gemma
```

You may instead fine-tune either model on the training split (requires substantially more memory and accepted model access):

```bash
python scripts/train_pretrained.py --architecture causal --model meta-llama/Llama-3.2-1B --output-dir outputs/llama-finetuned --epochs 1 --batch-size 1 --gradient-checkpointing
python scripts/train_pretrained.py --architecture causal --model google/gemma-3-1b-it --output-dir outputs/gemma-finetuned --epochs 1 --batch-size 1 --gradient-checkpointing
```

The causal training script masks the English instruction and source from the language-model loss so that the model learns to produce the Hindi target. Prefer the prompting path when compute is limited. Follow each model's license and access requirements.

### 5. Evaluate all four systems

```bash
python scripts/evaluate.py --architecture seq2seq --model outputs/basic-transformer --output-dir outputs/evaluation/basic
python scripts/evaluate.py --architecture seq2seq --model outputs/mt5-small --output-dir outputs/evaluation/mt5
python scripts/evaluate.py --architecture causal --model meta-llama/Llama-3.2-1B --output-dir outputs/evaluation/llama
python scripts/evaluate.py --architecture causal --model google/gemma-3-1b-it --output-dir outputs/evaluation/gemma
```

Each evaluation run writes predictions as CSV and metrics as JSON. Metrics include corpus BLEU, chrF, total and per-example inference time, parameter count, device, and temperature. Training scripts also save training runtime and parameter count. For a fair runtime comparison, use the same machine, test rows, batch size, and generation limit.

### 6. Analyze sampling temperature

The evaluator uses deterministic decoding when temperature is zero. For causal models, repeat evaluation at several positive temperatures, for example:

```bash
python scripts/evaluate.py --architecture causal --model meta-llama/Llama-3.2-1B --temperature 0.3 --output-dir outputs/evaluation/llama-t03
python scripts/evaluate.py --architecture causal --model meta-llama/Llama-3.2-1B --temperature 0.7 --output-dir outputs/evaluation/llama-t07
python scripts/evaluate.py --architecture causal --model google/gemma-3-1b-it --temperature 0.3 --output-dir outputs/evaluation/gemma-t03
python scripts/evaluate.py --architecture causal --model google/gemma-3-1b-it --temperature 0.7 --output-dir outputs/evaluation/gemma-t07
```

Compare BLEU/chrF and inspect sample translations for changes in fluency, consistency, and faithfulness. Record all generation settings. Sampling makes outputs stochastic; rerun with multiple seeds or report that limitation.

## Results

No results have been produced yet. Fill this table only after all runs finish. Include the exact checkpoint, subset size, hardware, batch size, decoding settings, and metric versions in the report.

| Model | BLEU | chrF | Train / fine-tune time | Inference time | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic Transformer | Pending | Pending | Pending | Pending | Pending |
| mT5 | Pending | Pending | Pending | Pending | Pending |
| LLaMA | Pending | Pending | N/A if prompted | Pending | Pending |
| Gemma | Pending | Pending | N/A if prompted | Pending | Pending |

## Responsible use

Inspect examples from every model. Translation systems can omit or add meaning, reflect dataset biases, misrepresent cultural context, or mishandle sensitive text. Use fluent human review for consequential or sensitive translations, document limitations, and do not treat generated output as verified fact.

## Repository layout

- `scripts/prepare_data.py` — download and export a seeded, bounded dataset subset.
- `scripts/train_basic_transformer.py` — train the encoder-decoder baseline from random initialization.
- `scripts/train_pretrained.py` — fine-tune an encoder-decoder or causal model.
- `scripts/evaluate.py` — generate translations, save predictions, and calculate metrics.
- `requirements.txt` — Python dependencies.
- `data/processed/` and `outputs/` — generated locally; do not commit model weights or dataset files.

## Remaining assignment work

- [ ] Run dataset preparation and record actual subset counts and preprocessing.
- [ ] Train the basic Transformer and fine-tune mT5.
- [ ] Prompt or fine-tune LLaMA and Gemma.
- [ ] Evaluate all models on the exact same test file and compare temperature settings.
- [ ] Review translation examples and write the model comparison, responsible-use discussion, and evidence-based conclusion.
- [ ] Verify dataset and model license/access requirements and capture the software/hardware environment.

