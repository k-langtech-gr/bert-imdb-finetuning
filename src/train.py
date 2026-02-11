import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .data import load_imdb
from .metrics import compute_metrics


@dataclass
class Config:
    model_ckpt: str
    max_len: int = 256

    # split
    val_size: float = 0.1
    seed: int = 42

    # training
    output_dir: str = "runs/model"
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"

    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    fp16: bool = True
    report_to: str = "none"


def _read_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return Config(**raw)


def _tokenize(ds, tokenizer, max_len: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)
    return ds.map(tok, batched=True, remove_columns=["text"])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer model on IMDb sentiment classification.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    cfg = _read_config(args.config)

    imdb = load_imdb(test_size=cfg.val_size, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckpt, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_tok = _tokenize(imdb["train"], tokenizer, cfg.max_len)
    val_tok   = _tokenize(imdb["val"], tokenizer, cfg.max_len)
    test_tok  = _tokenize(imdb["test"], tokenizer, cfg.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_ckpt, num_labels=2)

    os.makedirs(cfg.output_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        report_to=cfg.report_to,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(test_tok)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
