from __future__ import annotations

import gc
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import re

import torch
from tqdm.auto import tqdm

from .config import save_yaml
from .data import build_dataloaders
from .model import build_model_from_config
from .optimizers import TraceableOptimizer, build_optimizer
from .utils import (
    append_jsonl,
    bytes_to_gb,
    configure_torch_backends,
    count_parameters,
    ensure_dir,
    optimizer_state_bytes,
    resolve_device,
    set_seed,
    tensor_bytes,
    write_json,
)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    model = build_model_from_config(model_cfg)
    print(f"Model parameters: {count_parameters(model):,}")
    return model


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    train_cfg = config["training"]
    total_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]
    min_ratio = train_cfg["min_lr_ratio"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        with _autocast_context(device, precision):
            losses.append(float(model(**batch)["loss"].detach().cpu()))
    loss = float(sum(losses) / max(len(losses), 1))
    return {"eval_loss": loss, "eval_perplexity": math.exp(min(20.0, loss))}


def compute_directional_sharpness(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
) -> float:
    del precision
    model.eval()
    model.zero_grad(set_to_none=True)
    params = [param for param in model.parameters() if param.requires_grad]
    loss = model(**batch)["loss"]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = _flatten_tensors([grad.float() for grad in grads])
    direction = flat_grad / (flat_grad.norm() + 1e-12)

    start = 0
    directional_grad = 0.0
    for grad in grads:
        numel = grad.numel()
        directional_grad = directional_grad + (
            grad.reshape(-1).float() * direction[start : start + numel]
        ).sum()
        start += numel
    hvps = torch.autograd.grad(directional_grad, params, retain_graph=False)
    flat_hvp = _flatten_tensors([hvp.float() for hvp in hvps])
    sharpness = torch.dot(direction, flat_hvp).item()
    model.zero_grad(set_to_none=True)
    return float(sharpness)


def profile_memory_breakdown(
    model: torch.nn.Module,
    optimizer: TraceableOptimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "weights_gb": 0.0,
            "gradients_gb": 0.0,
            "optimizer_states_gb": 0.0,
            "activations_gb": 0.0,
            "peak_memory_gb": 0.0,
        }

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    weights_bytes = sum(tensor_bytes(param) for param in model.parameters())
    with _autocast_context(device, precision):
        loss = model(**batch)["loss"]
    loss.backward()
    optimizer.step()
    grads_bytes = sum(tensor_bytes(param.grad) for param in model.parameters() if param.grad is not None)
    opt_state_bytes = optimizer_state_bytes(optimizer)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    activations_bytes = max(0, peak_bytes - weights_bytes - grads_bytes - opt_state_bytes)
    return {
        "weights_gb": bytes_to_gb(weights_bytes),
        "gradients_gb": bytes_to_gb(grads_bytes),
        "optimizer_states_gb": bytes_to_gb(opt_state_bytes),
        "activations_gb": bytes_to_gb(activations_bytes),
        "peak_memory_gb": bytes_to_gb(peak_bytes),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_block(device: torch.device, fn) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end) / 1000.0
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _resolve_resume_checkpoint(output_dir: Path, resume: str | None) -> Path | None:
    if resume is None or str(resume).lower() in {"none", "false", ""}:
        return None
    if str(resume).lower() == "latest":
        candidates = []
        for path in output_dir.glob("checkpoint_step_*.pt"):
            match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]
    return Path(resume)


def train_experiment(config: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    configure_torch_backends()
    set_seed(config["seed"])
    device = resolve_device(config["training"].get("device", "auto"))

    output_dir = ensure_dir(Path(config["output_root"]) / config["experiment_name"])
    save_yaml(output_dir / "config.yaml", config)
    metrics_path = output_dir / "metrics.jsonl"
    scaling_path = output_dir / "scaling_traces.jsonl"
    sharpness_path = output_dir / "sharpness.jsonl"
    checkpoint_path = _resolve_resume_checkpoint(output_dir, resume)
    is_resuming = checkpoint_path is not None and checkpoint_path.exists()
    if not is_resuming:
        if metrics_path.exists():
            metrics_path.unlink()
        if scaling_path.exists():
            scaling_path.unlink()
        if sharpness_path.exists():
            sharpness_path.unlink()

    train_loader, eval_loader, _ = build_dataloaders(config)
    model = build_model(config).to(device)
    if config["training"].get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    precision = config["training"]["mixed_precision"]
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    train_iterator = _infinite_loader(train_loader)
    train_cfg = config["training"]
    analysis_cfg = config["analysis"]
    start_step = 0

    summary = {
        "experiment_name": config["experiment_name"],
        "device": str(device),
        "parameter_count": count_parameters(model),
        "best_eval_loss": None,
        "best_eval_perplexity": None,
        "best_eval_step": None,
    }

    if is_resuming:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if use_scaler and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", 0))
        saved_summary = checkpoint.get("summary")
        if isinstance(saved_summary, dict):
            summary.update(saved_summary)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(
        range(start_step + 1, train_cfg["max_steps"] + 1),
        desc=config["experiment_name"],
        initial=start_step,
        total=train_cfg["max_steps"],
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        backward_time = 0.0

        for _ in range(train_cfg["grad_accum_steps"]):
            batch = next(train_iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast_context(device, precision):
                loss = model(**batch)["loss"] / train_cfg["grad_accum_steps"]
            def backward_call():
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            backward_time += _time_block(device, backward_call)
            running_loss += float(loss.detach().cpu())

        if train_cfg["max_grad_norm"] is not None:
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"]))
        else:
            grad_norm = 0.0

        def optimizer_call():
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        optimizer_step_time = _time_block(device, optimizer_call)
        scheduler.step()

        if analysis_cfg.get("enable_scaling_trace", False):
            for trace in optimizer.pop_traces():
                append_jsonl(scaling_path, trace)

        if analysis_cfg.get("enable_sharpness_online", False) and step in set(analysis_cfg.get("sharpness_steps", [])):
            sharpness_batch = next(train_iterator)
            sharpness_batch = {key: value.to(device) for key, value in sharpness_batch.items()}
            sharpness = compute_directional_sharpness(model, sharpness_batch, device, precision)
            append_jsonl(sharpness_path, {"step": step, "sharpness": sharpness})

        if step % train_cfg["log_interval"] == 0 or step == 1:
            payload = {
                "step": step,
                "train_loss": running_loss,
                "train_perplexity": math.exp(min(20.0, running_loss)),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
            }
            if analysis_cfg.get("enable_step_timing", False) and step <= train_cfg["timing_window_steps"]:
                payload["backward_time_s"] = backward_time
                payload["optimizer_step_time_s"] = optimizer_step_time
                payload["step_time_s"] = backward_time + optimizer_step_time
            if device.type == "cuda":
                payload["peak_memory_gb"] = bytes_to_gb(torch.cuda.max_memory_allocated(device))
            append_jsonl(metrics_path, payload)
            progress.set_postfix(loss=f"{running_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if step % train_cfg["eval_interval"] == 0 or step == train_cfg["max_steps"]:
            eval_metrics = evaluate(model, eval_loader, device, precision, train_cfg["eval_batches"])
            append_jsonl(metrics_path, {"step": step, **eval_metrics})
            if summary["best_eval_loss"] is None or eval_metrics["eval_loss"] < summary["best_eval_loss"]:
                summary["best_eval_loss"] = eval_metrics["eval_loss"]
                summary["best_eval_perplexity"] = eval_metrics["eval_perplexity"]
                summary["best_eval_step"] = step
                torch.save(model.state_dict(), output_dir / "best_model.pt")

        if step in set(train_cfg.get("save_analysis_steps", [])):
            torch.save(model.state_dict(), output_dir / f"analysis_step_{step}.pt")

        if step % train_cfg["save_interval"] == 0 or step == train_cfg["max_steps"]:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else None,
                    "step": step,
                    "config": config,
                    "summary": summary,
                },
                output_dir / f"checkpoint_step_{step}.pt",
            )

    summary["final_learning_rate"] = scheduler.get_last_lr()[0]
    if device.type == "cuda":
        summary["peak_memory_gb"] = bytes_to_gb(torch.cuda.max_memory_allocated(device))
    write_json(output_dir / "summary.json", summary)

    del model, optimizer, scheduler, train_loader, eval_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
