from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import save_yaml
from .optimizers import build_optimizer
from .train import _autocast_context, _build_scheduler, _resolve_resume_checkpoint, build_model
from .utils import append_jsonl, bytes_to_gb, ensure_dir, resolve_device, set_seed, write_json


_ARC_TASKS = {
    "arc_easy": {"dataset_name": "allenai/ai2_arc", "config_name": "ARC-Easy"},
    "arc_challenge": {"dataset_name": "allenai/ai2_arc", "config_name": "ARC-Challenge"},
}


def _normalize_answer_key(answer_key: str | None) -> str | None:
    if answer_key is None:
        return None
    value = str(answer_key).strip()
    digit_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return digit_map.get(value, value)


def _build_arc_prompt(example: dict[str, Any]) -> tuple[str, list[str], int]:
    question = example["question"]
    choice_labels = example["choices"]["label"]
    choice_texts = example["choices"]["text"]
    pairs = list(zip(choice_labels, choice_texts))
    prompt_lines = ["Question:", question, "", "Choices:"]
    for label, text in pairs:
        prompt_lines.append(f"{label}. {text}")
    prompt_lines.extend(["", "Answer:"])
    prompt = "\n".join(prompt_lines)
    answer_key = _normalize_answer_key(example.get("answerKey"))
    answer_index = next((idx for idx, (label, _) in enumerate(pairs) if label == answer_key), -1)
    return prompt, [text for _, text in pairs], answer_index


def _prepare_arc_splits(config: dict[str, Any]):
    task_cfg = config["task"]
    task_name = str(task_cfg["name"]).lower()
    if task_name not in _ARC_TASKS:
        raise ValueError(f"Unsupported finetune task: {task_cfg['name']}")

    dataset_name = task_cfg.get("dataset_name", _ARC_TASKS[task_name]["dataset_name"])
    config_name = task_cfg.get("config_name", _ARC_TASKS[task_name]["config_name"])
    cache_dir = config["dataset"].get("cache_dir")

    train_split = load_dataset(dataset_name, config_name, split=task_cfg.get("train_split", "train"), cache_dir=cache_dir)
    eval_split = load_dataset(dataset_name, config_name, split=task_cfg.get("eval_split", "validation"), cache_dir=cache_dir)

    max_train = task_cfg.get("max_train_examples")
    max_eval = task_cfg.get("max_eval_examples")
    if max_train is not None:
        train_split = train_split.select(range(min(max_train, len(train_split))))
    if max_eval is not None:
        eval_split = eval_split.select(range(min(max_eval, len(eval_split))))
    return train_split, eval_split


def _build_tokenizer(config: dict[str, Any]):
    ds_cfg = config["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(
        ds_cfg["tokenizer_name"],
        use_fast=True,
        cache_dir=ds_cfg.get("cache_dir"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class MultipleChoiceTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, sequence_length: int) -> None:
        self.rows: list[dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or ""
        ignore_index = -100
        for example in dataset:
            prompt, choices, answer_index = _build_arc_prompt(example)
            if answer_index < 0:
                continue
            answer_text = " " + choices[answer_index] + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + answer_ids)[:sequence_length]
            labels = ([ignore_index] * len(prompt_ids) + answer_ids)[:sequence_length]
            attention_mask = [1] * len(input_ids)
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


class MultipleChoiceEvalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        self.rows: list[dict[str, Any]] = []
        for example in dataset:
            prompt, choices, answer_index = _build_arc_prompt(example)
            if answer_index < 0:
                continue
            self.rows.append({"prompt": prompt, "choices": choices, "label": answer_index})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _pad_train_collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
            labels.append(F.pad(item["labels"], (0, pad_len), value=-100))
            attention_mask.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }

    return collate_fn


def _score_choice(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    choice_text: str,
    device: torch.device,
    precision: str,
    sequence_length: int,
) -> tuple[float, float]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    choice_ids = tokenizer(" " + choice_text, add_special_tokens=False)["input_ids"]
    if not choice_ids:
        return float("-inf"), float("inf")
    input_ids = (prompt_ids + choice_ids)[:sequence_length]
    prompt_len = min(len(prompt_ids), max(0, len(input_ids) - 1))
    tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with _autocast_context(device, precision):
        logits = model(input_ids=tensor)["logits"]
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = tensor[:, 1:]
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    choice_token_scores = token_log_probs[:, prompt_len:]
    if choice_token_scores.numel() == 0:
        return float("-inf"), float("inf")
    mean_log_prob = float(choice_token_scores.mean().item())
    neg_log_likelihood = float(-choice_token_scores.sum().item())
    return mean_log_prob, neg_log_likelihood


@torch.no_grad()
def evaluate_multiple_choice(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    tokenizer,
    device: torch.device,
    precision: str,
    sequence_length: int,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    losses = []
    for batch in eval_loader:
        prompts = batch["prompt"]
        choice_batches = batch["choices"]
        labels = batch["label"]
        batch_size = len(prompts)
        for idx in range(batch_size):
            prompt = prompts[idx]
            choices = choice_batches[idx]
            label = int(labels[idx])
            scored = [
                _score_choice(model, tokenizer, prompt, choice, device, precision, sequence_length)
                for choice in choices
            ]
            best_index = max(range(len(scored)), key=lambda i: scored[i][0])
            losses.append(scored[label][1])
            correct += int(best_index == label)
            total += 1
    avg_loss = float(sum(losses) / max(len(losses), 1))
    return {"eval_loss": avg_loss, "eval_accuracy": correct / max(total, 1)}


def _build_finetune_loaders(config: dict[str, Any]):
    tokenizer = _build_tokenizer(config)
    train_split, eval_split = _prepare_arc_splits(config)
    sequence_length = int(config["dataset"]["sequence_length"])

    train_dataset = MultipleChoiceTrainDataset(train_split, tokenizer, sequence_length)
    eval_dataset = MultipleChoiceEvalDataset(eval_split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_pad_train_collate(tokenizer),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.get("task", {}).get("eval_batch_size", 4),
        shuffle=False,
        num_workers=config["dataset"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda rows: {
            "prompt": [row["prompt"] for row in rows],
            "choices": [row["choices"] for row in rows],
            "label": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        },
    )
    return train_loader, eval_loader, tokenizer


def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def finetune_experiment(config: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    del resume
    set_seed(config["seed"])
    device = resolve_device(config["training"].get("device", "auto"))
    output_dir = ensure_dir(Path(config["output_root"]) / config["experiment_name"])
    save_yaml(output_dir / "config.yaml", config)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    train_loader, eval_loader, tokenizer = _build_finetune_loaders(config)
    train_iterator = _infinite_loader(train_loader)
    train_cfg = config["training"]
    steps_per_epoch = max(1, len(train_loader))
    epochs = int(train_cfg.get("epochs", 1))
    explicit_max_steps = train_cfg.get("max_steps")
    total_steps = int(explicit_max_steps) if explicit_max_steps is not None else steps_per_epoch * epochs
    config["training"]["max_steps"] = total_steps

    model = build_model(config).to(device)
    init_checkpoint = config.get("task", {}).get("init_checkpoint")
    if init_checkpoint:
        checkpoint_path = Path(str(init_checkpoint))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    optimizer = build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    precision = train_cfg["mixed_precision"]
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best_eval_accuracy = None
    best_eval_loss = None
    best_eval_step = None

    progress = tqdm(range(1, total_steps + 1), desc=config["experiment_name"])
    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(train_cfg["grad_accum_steps"]):
            batch = next(train_iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast_context(device, precision):
                loss = model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"] / train_cfg["grad_accum_steps"]
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().cpu())

        if train_cfg["max_grad_norm"] is not None:
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"]))
        else:
            grad_norm = 0.0

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if step % train_cfg["log_interval"] == 0 or step == 1:
            append_jsonl(
                metrics_path,
                {
                    "step": step,
                    "epoch": (step - 1) / steps_per_epoch,
                    "train_loss": running_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                },
            )
            progress.set_postfix(loss=f"{running_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if step % train_cfg["eval_interval"] == 0 or step == total_steps:
            eval_metrics = evaluate_multiple_choice(
                model,
                eval_loader,
                tokenizer,
                device,
                precision,
                int(config["dataset"]["sequence_length"]),
            )
            append_jsonl(metrics_path, {"step": step, "epoch": step / steps_per_epoch, **eval_metrics})
            if best_eval_accuracy is None or eval_metrics["eval_accuracy"] > best_eval_accuracy:
                best_eval_accuracy = eval_metrics["eval_accuracy"]
                best_eval_loss = eval_metrics["eval_loss"]
                best_eval_step = step
                torch.save(model.state_dict(), output_dir / "best_model.pt")

    summary = {
        "experiment_name": config["experiment_name"],
        "task_name": config["task"]["name"],
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "best_eval_accuracy": best_eval_accuracy,
        "best_eval_loss": best_eval_loss,
        "best_eval_step": best_eval_step,
    }
    if device.type == "cuda":
        summary["peak_memory_gb"] = bytes_to_gb(torch.cuda.max_memory_allocated(device))
    write_json(output_dir / "summary.json", summary)

    del model, optimizer, scheduler, train_loader, eval_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
