from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _prepare_split(
    dataset_name: str,
    config_name: str | None,
    split: str,
    cache_dir: str | None,
) -> Dataset:
    return load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir)


def _select_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, Any]:
    ds_cfg = config["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(ds_cfg["tokenizer_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _prepare_split(
        ds_cfg["name"],
        ds_cfg.get("config_name"),
        ds_cfg["train_split"],
        ds_cfg.get("cache_dir"),
    )
    eval_dataset = _prepare_split(
        ds_cfg["name"],
        ds_cfg.get("config_name"),
        ds_cfg["eval_split"],
        ds_cfg.get("cache_dir"),
    )
    train_dataset = _select_subset(train_dataset, ds_cfg.get("max_train_examples"))
    eval_dataset = _select_subset(eval_dataset, ds_cfg.get("max_eval_examples"))

    text_field = ds_cfg["text_field"]
    seq_len = ds_cfg["sequence_length"]
    eos_id = tokenizer.eos_token_id
    num_proc = ds_cfg.get("num_proc", 1)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        cleaned = [text if text is not None else "" for text in batch[text_field]]
        tokenized = tokenizer(cleaned, add_special_tokens=False)
        tokenized["input_ids"] = [ids + [eos_id] for ids in tokenized["input_ids"]]
        return {"input_ids": tokenized["input_ids"]}

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        flat = list(chain.from_iterable(batch["input_ids"]))
        total = (len(flat) // seq_len) * seq_len
        chunks = [flat[i : i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": chunks, "labels": [chunk[:] for chunk in chunks]}

    train_tokenized = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing train split",
    )
    eval_tokenized = eval_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing eval split",
    )
    train_grouped = train_tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Packing train split",
    )
    eval_grouped = eval_tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Packing eval split",
    )

    train_grouped.set_format(type="torch", columns=["input_ids", "labels"])
    eval_grouped.set_format(type="torch", columns=["input_ids", "labels"])

    train_loader = DataLoader(
        train_grouped,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_grouped,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, eval_loader, tokenizer

