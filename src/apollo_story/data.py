from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from datasets import Dataset, IterableDataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _prepare_split(
    dataset_name: str,
    config_name: str | None,
    split: str,
    cache_dir: str | None,
    streaming: bool,
) -> Dataset | IterableDataset:
    return load_dataset(
        dataset_name,
        config_name,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )


def _select_subset(dataset: Dataset | IterableDataset, limit: int | None) -> Dataset | IterableDataset:
    if limit is None:
        return dataset
    if isinstance(dataset, IterableDataset):
        return dataset.take(limit)
    return dataset.select(range(min(limit, len(dataset))))


def _set_torch_format(dataset: Dataset | IterableDataset) -> Dataset | IterableDataset:
    if isinstance(dataset, IterableDataset):
        return dataset.with_format("torch")
    dataset.set_format(type="torch", columns=["input_ids", "labels"])
    return dataset


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, Any]:
    ds_cfg = config["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(
        ds_cfg["tokenizer_name"],
        use_fast=True,
        cache_dir=ds_cfg.get("cache_dir"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_streaming = bool(ds_cfg.get("streaming_train", False))
    eval_streaming = bool(ds_cfg.get("streaming_eval", False))

    train_dataset = _prepare_split(
        ds_cfg["name"],
        ds_cfg.get("config_name"),
        ds_cfg["train_split"],
        ds_cfg.get("cache_dir"),
        train_streaming,
    )
    eval_dataset = _prepare_split(
        ds_cfg["name"],
        ds_cfg.get("config_name"),
        ds_cfg["eval_split"],
        ds_cfg.get("cache_dir"),
        eval_streaming,
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

    train_map_kwargs = {
        "batched": True,
        "remove_columns": train_dataset.column_names,
        "desc": "Tokenizing train split",
    }
    eval_map_kwargs = {
        "batched": True,
        "remove_columns": eval_dataset.column_names,
        "desc": "Tokenizing eval split",
    }
    if not isinstance(train_dataset, IterableDataset):
        train_map_kwargs["num_proc"] = num_proc
    if not isinstance(eval_dataset, IterableDataset):
        eval_map_kwargs["num_proc"] = num_proc

    train_tokenized = train_dataset.map(tokenize_batch, **train_map_kwargs)
    eval_tokenized = eval_dataset.map(tokenize_batch, **eval_map_kwargs)

    pack_train_kwargs = {"batched": True, "desc": "Packing train split"}
    pack_eval_kwargs = {"batched": True, "desc": "Packing eval split"}
    if not isinstance(train_tokenized, IterableDataset):
        pack_train_kwargs["num_proc"] = num_proc
    if not isinstance(eval_tokenized, IterableDataset):
        pack_eval_kwargs["num_proc"] = num_proc

    train_grouped = train_tokenized.map(group_texts, **pack_train_kwargs)
    eval_grouped = eval_tokenized.map(group_texts, **pack_eval_kwargs)

    train_grouped = _set_torch_format(train_grouped)
    eval_grouped = _set_torch_format(eval_grouped)

    train_loader = DataLoader(
        train_grouped,
        batch_size=config["training"]["batch_size"],
        shuffle=not isinstance(train_grouped, IterableDataset),
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
