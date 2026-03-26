from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from .config import save_yaml
from .data import build_dataloaders
from .optimizers import build_optimizer
from .train import _autocast_context, build_model, profile_memory_breakdown
from .utils import append_jsonl, bytes_to_gb, ensure_dir, resolve_device, set_seed, write_json


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _capacity_limit_reason(exc: RuntimeError) -> str | None:
    message = str(exc).lower()
    if "out of memory" in message:
        return "oom"
    if "exceeded logical memory budget" in message:
        return "memory_budget"
    return None


def _is_capacity_limited(exc: RuntimeError) -> bool:
    return _capacity_limit_reason(exc) is not None


def _synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _event_timers(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None


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


def _measure_at_batch_size(
    config: dict[str, Any],
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    precision = config["training"]["mixed_precision"]
    synthetic = config["benchmark"].get("synthetic", True)
    step_trace_path = output_dir / "step_times.jsonl" if output_dir is not None else None

    if synthetic:
        def next_batch():
            return _synthetic_batch(
                batch_size,
                config["dataset"]["sequence_length"],
                config["model"]["vocab_size"],
                device,
            )
    else:
        train_loader, _, _ = build_dataloaders(config)
        iterator = iter(train_loader)

        def next_batch():
            nonlocal iterator
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            return {key: value.to(device) for key, value in batch.items()}

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(warmup_steps):
            batch = next_batch()
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, precision):
                loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()

        step_times = []
        total_tokens = 0
        total_elapsed = 0.0
        if step_trace_path is not None and step_trace_path.exists():
            step_trace_path.unlink()

        for step in range(1, measure_steps + 1):
            batch = next_batch()
            optimizer.zero_grad(set_to_none=True)

            loss_holder: dict[str, torch.Tensor] = {}

            def run_forward() -> None:
                with _autocast_context(device, precision):
                    loss_holder["loss"] = model(**batch)["loss"]

            forward_time = _time_block(device, run_forward)
            loss = loss_holder["loss"]

            backward_time = _time_block(device, loss.backward)
            opt_time = _time_block(device, optimizer.step)

            step_time = forward_time + backward_time + opt_time
            total_elapsed += step_time
            total_tokens += batch["input_ids"].numel()
            record = {
                "step": step,
                "forward_time_s": forward_time,
                "backward_time_s": backward_time,
                "optimizer_step_time_s": opt_time,
                "step_time_s": step_time,
            }
            step_times.append(record)
            if step_trace_path is not None:
                append_jsonl(step_trace_path, record)

        peak_memory_gb = bytes_to_gb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
        budget = config["benchmark"].get("memory_budget_gb")
        if budget is not None and peak_memory_gb > budget:
            raise RuntimeError(f"Exceeded logical memory budget: {peak_memory_gb:.2f} GB > {budget:.2f} GB")

        summary = {
            "batch_size": batch_size,
            "tokens_per_second": total_tokens / max(total_elapsed, 1e-12),
            "avg_step_time_s": total_elapsed / max(measure_steps, 1),
            "avg_forward_time_s": sum(x["forward_time_s"] for x in step_times) / max(measure_steps, 1),
            "peak_memory_gb": peak_memory_gb,
            "avg_backward_time_s": sum(x["backward_time_s"] for x in step_times) / max(measure_steps, 1),
            "avg_optimizer_step_time_s": sum(x["optimizer_step_time_s"] for x in step_times) / max(measure_steps, 1),
        }
        summary["avg_total_step_time_s"] = summary["avg_step_time_s"]

        if config["benchmark"].get("profile_memory", False):
            batch = next_batch()
            summary["memory_breakdown"] = profile_memory_breakdown(model, optimizer, batch, device, precision)

        del model, optimizer
        _cleanup(device)
        return summary
    except RuntimeError:
        _cleanup(device)
        raise


def _measure_batch_sweep(config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    bench_cfg = config["benchmark"]
    sweep_path = output_dir / "batch_sweep.jsonl"
    if sweep_path.exists():
        sweep_path.unlink()

    records: list[dict[str, Any]] = []
    current = bench_cfg["initial_batch_size"]
    cap = bench_cfg["batch_size_cap"]
    while current <= cap:
        try:
            summary = _measure_at_batch_size(config, current, warmup_steps=1, measure_steps=1)
            record = {
                "batch_size": current,
                "success": True,
                "peak_memory_gb": summary["peak_memory_gb"],
            }
            records.append(record)
            append_jsonl(sweep_path, record)
            current *= 2
        except RuntimeError as exc:
            if not _is_capacity_limited(exc):
                raise
            record = {
                "batch_size": current,
                "success": False,
                "peak_memory_gb": None,
                "failure_reason": _capacity_limit_reason(exc),
            }
            records.append(record)
            append_jsonl(sweep_path, record)
            break
    return records


def _find_max_batch_size(config: dict[str, Any], output_dir: Path) -> tuple[int, list[dict[str, Any]]]:
    records = _measure_batch_sweep(config, output_dir)
    successful = [row["batch_size"] for row in records if row["success"]]
    best = max(successful) if successful else 0

    low = best + 1
    current = max([row["batch_size"] for row in records], default=best)
    cap = config["benchmark"]["batch_size_cap"]
    high = min(cap, current - 1 if current > best else cap)
    sweep_path = output_dir / "batch_sweep.jsonl"
    while low <= high:
        mid = (low + high) // 2
        try:
            summary = _measure_at_batch_size(config, mid, warmup_steps=1, measure_steps=1)
            best = mid
            record = {
                "batch_size": mid,
                "success": True,
                "peak_memory_gb": summary["peak_memory_gb"],
            }
            records.append(record)
            append_jsonl(sweep_path, record)
            low = mid + 1
        except RuntimeError as exc:
            if not _is_capacity_limited(exc):
                raise
            record = {
                "batch_size": mid,
                "success": False,
                "peak_memory_gb": None,
                "failure_reason": _capacity_limit_reason(exc),
            }
            records.append(record)
            append_jsonl(sweep_path, record)
            high = mid - 1
    records = sorted(records, key=lambda row: row["batch_size"])
    return best, records


def _resolve_fixed_batch_size(config: dict[str, Any]) -> int:
    bench_cfg = config["benchmark"]
    fixed_batch = bench_cfg.get("fixed_batch_size")
    if fixed_batch is None:
        fixed_batch = config["training"]["batch_size"]
    fixed_batch = int(fixed_batch)
    if fixed_batch < 1:
        raise RuntimeError("Fixed batch size must be at least 1.")
    return fixed_batch


def benchmark_experiment(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config["seed"])
    device = resolve_device(config["training"].get("device", "auto"))
    output_dir = ensure_dir(Path(config["output_root"]) / config["experiment_name"])
    save_yaml(output_dir / "config.yaml", config)
    bench_cfg = config["benchmark"]
    benchmark_mode = bench_cfg.get("mode", "max_batch")

    if benchmark_mode == "max_batch":
        max_batch_size, sweep_records = _find_max_batch_size(config, output_dir)
        if max_batch_size < 1:
            raise RuntimeError("Could not fit even a single batch. Reduce the model or sequence length.")
        final_batch_size = max_batch_size
        while final_batch_size >= 1:
            try:
                summary = _measure_at_batch_size(
                    config,
                    batch_size=final_batch_size,
                    warmup_steps=bench_cfg["warmup_steps"],
                    measure_steps=bench_cfg["measure_steps"],
                    output_dir=output_dir,
                )
                break
            except RuntimeError as exc:
                if not _is_capacity_limited(exc) or final_batch_size == 1:
                    raise
                final_batch_size -= 1
        summary["max_batch_size"] = final_batch_size
        summary["batch_sweep"] = sweep_records
    elif benchmark_mode == "fixed_batch":
        fixed_batch_size = _resolve_fixed_batch_size(config)
        summary = _measure_at_batch_size(
            config,
            batch_size=fixed_batch_size,
            warmup_steps=bench_cfg["warmup_steps"],
            measure_steps=bench_cfg["measure_steps"],
            output_dir=output_dir,
        )
        summary["max_batch_size"] = None
        summary["batch_sweep"] = []
        summary["fixed_batch_size"] = fixed_batch_size
    else:
        raise RuntimeError(f"Unsupported benchmark mode: {benchmark_mode}")

    summary["benchmark_mode"] = benchmark_mode
    summary["measured_batch_size"] = summary["batch_size"]
    summary["experiment_name"] = config["experiment_name"]
    summary["device"] = str(device)
    summary["memory_budget_gb"] = bench_cfg.get("memory_budget_gb")
    memory_profile_batch_size = config["benchmark"].get("memory_profile_batch_size")
    if config["benchmark"].get("profile_memory", False):
        profile_batch_size = int(memory_profile_batch_size or summary["batch_size"])
        try:
            memory_summary = _measure_at_batch_size(
                config,
                batch_size=profile_batch_size,
                warmup_steps=max(1, min(bench_cfg["warmup_steps"], 3)),
                measure_steps=1,
                output_dir=None,
            )
            summary["memory_breakdown"] = memory_summary.get("memory_breakdown")
            summary["memory_profile_batch_size"] = profile_batch_size
        except RuntimeError as exc:
            if not _is_capacity_limited(exc):
                raise
            summary["memory_breakdown"] = None
            summary["memory_profile_batch_size"] = profile_batch_size
    write_json(output_dir / "benchmark_summary.json", summary)
    return summary
