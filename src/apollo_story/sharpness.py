from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import load_yaml
from .data import build_dataloaders
from .optimizers import build_optimizer
from .train import build_model, compute_directional_sharpness
from .utils import append_jsonl, resolve_device


def _truncate_batch(batch: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    return {key: value[:batch_size] for key, value in batch.items()}


def analyze_experiment(experiment_dir: str | Path) -> None:
    exp_dir = Path(experiment_dir)
    config = load_yaml(exp_dir / "config.yaml")
    device = resolve_device(config["training"].get("device", "auto"))
    _, eval_loader, _ = build_dataloaders(config)
    eval_iter = iter(eval_loader)
    precision = config["training"]["mixed_precision"]
    sharpness_batch_size = int(config.get("analysis", {}).get("sharpness_batch_size", 1))
    sharpness_num_batches = int(config.get("analysis", {}).get("sharpness_num_batches", 1))
    output_path = exp_dir / "sharpness_analysis.jsonl"
    if output_path.exists():
        output_path.unlink()

    for checkpoint in sorted(exp_dir.glob("analysis_step_*.pt")):
        model = build_model(config).to(device)
        optimizer = build_optimizer(model, config)
        payload = torch.load(checkpoint, map_location=device)
        if isinstance(payload, dict) and "model" in payload:
            model.load_state_dict(payload["model"])
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
                direction_source = "optimizer_update"
            else:
                optimizer = None
                direction_source = "gradient_fallback"
        else:
            model.load_state_dict(payload)
            optimizer = None
            direction_source = "gradient_fallback"

        values = []
        for _ in range(max(1, sharpness_num_batches)):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_loader)
                batch = next(eval_iter)
            batch = _truncate_batch(batch, sharpness_batch_size)
            batch = {key: value.to(device) for key, value in batch.items()}
            values.append(compute_directional_sharpness(model, batch, device, precision, optimizer=optimizer))

        sharpness = float(sum(values) / max(len(values), 1))
        step = int(checkpoint.stem.split("_")[-1])
        append_jsonl(
            output_path,
            {
                "step": step,
                "sharpness": sharpness,
                "direction_source": direction_source,
                "num_batches": max(1, sharpness_num_batches),
                "batch_size": sharpness_batch_size,
            },
        )
        del model, optimizer, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()


def analyze_suite(suite_dir: str | Path) -> None:
    suite_path = Path(suite_dir)
    for exp_dir in sorted(suite_path.iterdir()):
        if exp_dir.is_dir() and (exp_dir / "config.yaml").exists():
            analyze_experiment(exp_dir)
