# IMDb Sentiment Classification — BERT Fine-tuning

Fine-tuning **BERT** and **DistilBERT** for binary sentiment classification on the **IMDb** movie reviews dataset (Hugging Face `stanfordnlp/imdb`).

## What’s inside
- Reproducible data loading & cleaning (remove HTML `<br />`)
- Tokenization with a configurable `max_length`
- Training with `transformers.Trainer`
- Evaluation with `evaluate` (accuracy)
- Two example configs:
  - `google-bert/bert-base-uncased`
  - `distilbert/distilbert-base-uncased`

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train (BERT)
```bash
python -m imdb_bert_finetune.train --config configs/bert_base_uncased.yaml
```

### 3) Train (DistilBERT)
```bash
python -m imdb_bert_finetune.train --config configs/distilbert_base_uncased.yaml
```

Outputs (checkpoints, logs) are written under `runs/` by default.

## Notes
- Dataset: `stanfordnlp/imdb` from Hugging Face Datasets.
- Mixed precision (`fp16`) is enabled by default. If you train on CPU, set `fp16: false` in the config.
- The notebook version is preserved in `notebooks/`.

## Suggested citation / attribution
If you reuse the structure, please reference:
- Hugging Face Transformers & Datasets
- IMDb dataset (`stanfordnlp/imdb`)

