from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import load_yaml
from .data import build_dataloaders
from .train import _autocast_context, build_model, compute_directional_sharpness
from .utils import append_jsonl, ensure_dir, resolve_device


def analyze_experiment(experiment_dir: str | Path) -> None:
    exp_dir = Path(experiment_dir)
    config = load_yaml(exp_dir / "config.yaml")
    device = resolve_device(config["training"].get("device", "auto"))
    _, eval_loader, _ = build_dataloaders(config)
    eval_iter = iter(eval_loader)
    precision = config["training"]["mixed_precision"]
    output_path = exp_dir / "sharpness_analysis.jsonl"
    if output_path.exists():
        output_path.unlink()

    for checkpoint in sorted(exp_dir.glob("analysis_step_*.pt")):
        model = build_model(config).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        batch = next(eval_iter)
        batch = {key: value.to(device) for key, value in batch.items()}
        sharpness = compute_directional_sharpness(model, batch, device, precision)
        step = int(checkpoint.stem.split("_")[-1])
        append_jsonl(output_path, {"step": step, "sharpness": sharpness})


def analyze_suite(suite_dir: str | Path) -> None:
    suite_path = Path(suite_dir)
    for exp_dir in sorted(suite_path.iterdir()):
        if exp_dir.is_dir() and (exp_dir / "config.yaml").exists():
            analyze_experiment(exp_dir)

