from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
if os.environ.get("MPLBACKEND") == "module://matplotlib_inline.backend_inline":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _canonical_experiment_name(name: str) -> str:
    canonical = str(name)
    for suffix in ("_bench", "_oom_story"):
        if canonical.endswith(suffix):
            canonical = canonical[: -len(suffix)]
    return canonical


def _as_list(values: object) -> list[float]:
    if isinstance(values, list):
        return [float(value) for value in values]
    return [float(values)]


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _collect_summaries(result_dir: Path, summary_name: str) -> pd.DataFrame:
    rows = []
    for summary_path in result_dir.glob(f"*/{summary_name}"):
        with summary_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["experiment_dir"] = str(summary_path.parent)
        rows.append(row)
    return pd.DataFrame(rows)


def _collect_benchmark_summaries(micro_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(micro_dir.glob("*/benchmark_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["label"] = _canonical_experiment_name(row["experiment_name"])
        rows.append(row)
    return pd.DataFrame(rows)


def _shared_non_null_value(frame: pd.DataFrame, column: str) -> float | int | None:
    if column not in frame.columns:
        return None
    values = [value for value in frame[column].dropna().tolist()]
    if not values:
        return None
    if len(set(values)) == 1:
        return values[0]
    return None


def _final_eval_loss(metrics_path: Path) -> float | None:
    frame = _read_jsonl(metrics_path)
    if "eval_loss" not in frame.columns:
        return None
    eval_frame = frame.dropna(subset=["eval_loss"])
    if eval_frame.empty:
        return None
    return float(eval_frame.iloc[-1]["eval_loss"])


def plot_convergence(full_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for metrics_path in sorted(full_dir.glob("*/metrics.jsonl")):
        frame = _read_jsonl(metrics_path)
        if "train_loss" not in frame.columns:
            continue
        train_frame = frame.dropna(subset=["train_loss"])
        ax.plot(train_frame["step"], train_frame["train_loss"], label=metrics_path.parent.name)
    ax.set_title("The Convergence Race")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_validation_convergence(full_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for metrics_path in sorted(full_dir.glob("*/metrics.jsonl")):
        frame = _read_jsonl(metrics_path)
        if "eval_loss" not in frame.columns:
            continue
        eval_frame = frame.dropna(subset=["eval_loss"])
        if eval_frame.empty:
            continue
        plotted = True
        ax.plot(eval_frame["step"], eval_frame["eval_loss"], marker="o", label=metrics_path.parent.name)
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Validation Loss vs Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_finetune_accuracy(finetune_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for metrics_path in sorted(finetune_dir.glob("*/metrics.jsonl")):
        frame = _read_jsonl(metrics_path)
        if "eval_accuracy" not in frame.columns:
            continue
        eval_frame = frame.dropna(subset=["eval_accuracy"])
        if eval_frame.empty:
            continue
        plotted = True
        ax.plot(eval_frame["step"], eval_frame["eval_accuracy"], marker="o", label=metrics_path.parent.name)
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Validation Accuracy vs Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_finetune_table(finetune_dir: Path, output_csv: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(finetune_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_path = exp_dir / "summary.json"
        metrics_path = exp_dir / "metrics.jsonl"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "experiment_name": summary["experiment_name"],
                "task_name": summary.get("task_name"),
                "best_eval_accuracy": summary.get("best_eval_accuracy"),
                "best_eval_loss": summary.get("best_eval_loss"),
                "final_eval_loss": _final_eval_loss(metrics_path),
                "peak_memory_gb": summary.get("peak_memory_gb"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame.to_csv(output_csv, index=False)
        return frame
    frame = frame.sort_values(["task_name", "experiment_name"])
    frame.to_csv(output_csv, index=False)
    return frame


def make_table2_like(full_dir: Path, output_csv: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(full_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_path = exp_dir / "summary.json"
        metrics_path = exp_dir / "metrics.jsonl"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "experiment_name": summary["experiment_name"],
                "best_eval_perplexity": summary.get("best_eval_perplexity"),
                "best_eval_loss": summary.get("best_eval_loss"),
                "final_eval_loss": _final_eval_loss(metrics_path),
                "peak_memory_gb": summary.get("peak_memory_gb"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame.to_csv(output_csv, index=False)
        return frame
    frame = frame.sort_values("experiment_name")
    frame.to_csv(output_csv, index=False)
    return frame


def _fixed_batch_memory_table(micro_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(micro_dir.glob("*/benchmark_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        breakdown = summary.get("memory_breakdown") or {}
        fixed_batch_peak = breakdown.get("peak_memory_gb")
        fixed_batch_size = summary.get("memory_profile_batch_size")
        if fixed_batch_peak is None or fixed_batch_size is None:
            continue
        rows.append(
            {
                "experiment_name": _canonical_experiment_name(summary["experiment_name"]),
                "fixed_batch_peak_memory_gb": fixed_batch_peak,
                "memory_profile_batch_size": fixed_batch_size,
            }
        )
    return pd.DataFrame(rows)


def _merge_fixed_batch_memory(table_df: pd.DataFrame, micro_dir: Path | None) -> tuple[pd.DataFrame, str]:
    if micro_dir is None or not micro_dir.exists():
        return table_df, "Peak Memory (GB)"
    memory_df = _fixed_batch_memory_table(micro_dir)
    if memory_df.empty:
        return table_df, "Peak Memory (GB)"

    batch_sizes = set(memory_df["memory_profile_batch_size"].dropna().tolist())
    merged = table_df.merge(memory_df, on="experiment_name", how="left")
    if merged["fixed_batch_peak_memory_gb"].notna().any():
        merged["peak_memory_gb"] = merged["fixed_batch_peak_memory_gb"].combine_first(merged["peak_memory_gb"])
        if len(batch_sizes) == 1:
            batch_size = int(next(iter(batch_sizes)))
            return merged, f"Fixed-Batch Peak Memory (GB, batch={batch_size})"
        return merged, "Fixed-Batch Peak Memory (GB)"
    return table_df, "Peak Memory (GB)"


def plot_pareto(table_df: pd.DataFrame, output_path: Path, x_label: str = "Peak Memory (GB)") -> None:
    plot_df = table_df.dropna(subset=["peak_memory_gb", "final_eval_loss"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_df["peak_memory_gb"], plot_df["final_eval_loss"], s=180, c="#d1495b")
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["experiment_name"],
            (row["peak_memory_gb"], row["final_eval_loss"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Pareto Frontier: Memory vs Final Validation Loss")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Final Validation Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_scaling_ratio(full_dir: Path, output_path: Path) -> None:
    reference_name = "elementwise" if (full_dir / "elementwise" / "scaling_traces.jsonl").exists() else "adamw"
    reference_path = full_dir / reference_name / "scaling_traces.jsonl"
    if not reference_path.exists():
        return

    reference = _read_jsonl(reference_path)
    rows = []
    for trace_path in sorted(full_dir.glob("*/scaling_traces.jsonl")):
        label = trace_path.parent.name
        if label == reference_name:
            continue
        frame = _read_jsonl(trace_path)
        for _, row in frame.iterrows():
            matched = reference[
                (reference["step"] == row["step"])
                & (reference["param_name"] == row["param_name"])
            ]
            if matched.empty:
                continue
            base = _as_list(matched.iloc[0]["values"])
            target = _as_list(row["values"])
            size = min(len(base), len(target))
            ratios = [target[i] / max(base[i], 1e-12) for i in range(size)]
            for ratio in ratios:
                rows.append({"optimizer": label, "ratio": ratio, "step": row["step"]})

    if not rows:
        return
    ratio_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(ratio_df, x="optimizer", y="ratio", ax=ax, color="#66a182")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"Scaling Factor Ratio vs {reference_name}")
    ax.set_ylabel(f"Scale / {reference_name} scale")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_figure4_scaling(full_dir: Path, output_path: Path) -> None:
    reference_path = full_dir / "adamw" / "scaling_traces.jsonl"
    rank_1_4_path = full_dir / "apollo_rank_1_4" / "scaling_traces.jsonl"
    rank_1_8_path = full_dir / "apollo_rank_1_8" / "scaling_traces.jsonl"
    if not (reference_path.exists() and rank_1_4_path.exists() and rank_1_8_path.exists()):
        return

    reference = _read_jsonl(reference_path)
    rank_1_4 = _read_jsonl(rank_1_4_path)
    rank_1_8 = _read_jsonl(rank_1_8_path)
    if reference.empty or rank_1_4.empty or rank_1_8.empty:
        return

    common_steps = sorted(set(reference["step"]) & set(rank_1_4["step"]) & set(rank_1_8["step"]))
    if not common_steps:
        return
    target_step = common_steps[-1]
    param_names = reference["param_name"].drop_duplicates().tolist()[:3]
    if not param_names:
        return

    fig, axes = plt.subplots(1, len(param_names), figsize=(6 * len(param_names), 4.8), sharey=True)
    if len(param_names) == 1:
        axes = [axes]

    for ax, param_name in zip(axes, param_names):
        base_match = reference[(reference["step"] == target_step) & (reference["param_name"] == param_name)]
        r14_match = rank_1_4[(rank_1_4["step"] == target_step) & (rank_1_4["param_name"] == param_name)]
        r18_match = rank_1_8[(rank_1_8["step"] == target_step) & (rank_1_8["param_name"] == param_name)]
        if base_match.empty or r14_match.empty or r18_match.empty:
            continue
        base = _as_list(base_match.iloc[0]["values"])
        r14 = _as_list(r14_match.iloc[0]["values"])
        r18 = _as_list(r18_match.iloc[0]["values"])
        size = min(len(base), len(r14), len(r18))
        channels = list(range(size))
        ratio_14 = [r14[i] / max(base[i], 1e-12) for i in range(size)]
        ratio_18 = [r18[i] / max(base[i], 1e-12) for i in range(size)]

        ax.plot(channels, ratio_14, label="APOLLO 1/4n", linewidth=2)
        ax.plot(channels, ratio_18, label="APOLLO 1/8n", linewidth=2)
        ax.axhline(0.5, linestyle="--", color="#444444", linewidth=1.2, label="Theory 1/2 (1/4n)" if ax is axes[0] else None)
        ax.axhline(
            1.0 / math.sqrt(8.0),
            linestyle=":",
            color="#aa0000",
            linewidth=1.2,
            label=r"Theory $1/\sqrt{8}$ (1/8n)" if ax is axes[0] else None,
        )
        ax.set_title(param_name)
        ax.set_xlabel("Channel Index")
        ax.set_ylim(bottom=0.0)

    axes[0].set_ylabel("Scaling Ratio vs AdamW")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=9)
    fig.suptitle(f"Figure 4 Style Scaling Ratio Check (step={target_step})", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_memory_breakdown(micro_dir: Path, output_path: Path) -> None:
    rows = []
    batch_sizes = set()
    for summary_path in sorted(micro_dir.glob("*/benchmark_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        breakdown = summary.get("memory_breakdown")
        if not breakdown:
            continue
        profile_batch_size = summary.get("memory_profile_batch_size")
        if profile_batch_size is not None:
            batch_sizes.add(profile_batch_size)
        breakdown["experiment_name"] = _canonical_experiment_name(summary["experiment_name"])
        rows.append(breakdown)
    if not rows:
        return
    frame = pd.DataFrame(rows).set_index("experiment_name")
    components = ["weights_gb", "activations_gb", "gradients_gb", "optimizer_states_gb"]
    ax = frame[components].plot(kind="bar", stacked=True, figsize=(11, 6))
    if len(batch_sizes) == 1:
        batch_size = next(iter(batch_sizes))
        ax.set_title(f"Fixed-Batch Memory Breakdown (batch={batch_size})")
    else:
        ax.set_title("Fixed-Batch Memory Breakdown")
    ax.set_xlabel("")
    ax.set_ylabel("GB")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_batch_throughput(micro_dir: Path, output_path: Path) -> None:
    frame = _collect_benchmark_summaries(micro_dir)
    if frame.empty:
        return
    if "benchmark_mode" not in frame.columns:
        frame["benchmark_mode"] = "max_batch"
    if "measured_batch_size" not in frame.columns:
        if "batch_size" in frame.columns:
            frame["measured_batch_size"] = frame["batch_size"]
        elif "max_batch_size" in frame.columns:
            frame["measured_batch_size"] = frame["max_batch_size"]
        else:
            frame["measured_batch_size"] = None

    mode = _shared_non_null_value(frame, "benchmark_mode")
    budget = _shared_non_null_value(frame, "memory_budget_gb")
    fixed_batch = _shared_non_null_value(frame, "measured_batch_size")

    if mode == "fixed_batch":
        frame = frame.sort_values("tokens_per_second", ascending=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(frame, x="label", y="tokens_per_second", ax=ax, color="#00798c")
        ax.set_ylabel("Tokens / second")
        ax.set_xlabel("")
        title = "Throughput at Shared Fixed Batch"
        if fixed_batch is not None:
            title = f"Throughput at Shared Fixed Batch (batch={int(fixed_batch)})"
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    else:
        sort_column = "max_batch_size" if "max_batch_size" in frame.columns else "measured_batch_size"
        frame = frame.sort_values(sort_column, ascending=False)
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax2 = ax1.twinx()
        ax1.bar(frame["label"], frame["measured_batch_size"], color="#edae49", alpha=0.8)
        ax2.plot(frame["label"], frame["tokens_per_second"], color="#00798c", marker="o", linewidth=3)
        ax1.set_ylabel("Measured Batch Size")
        ax2.set_ylabel("Tokens / second")
        if budget is not None:
            ax1.set_title(f"Throughput under Shared Memory Budget ({float(budget):.2f} GB)")
        else:
            ax1.set_title("Max Batch Size and Throughput at Each Optimizer's Own Max Batch")
        ax1.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_budgeted_scaling_curves(micro_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for summary_path in sorted(micro_dir.glob("*/benchmark_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        sweep = summary.get("batch_sweep", [])
        if not sweep:
            continue
        success_df = pd.DataFrame([row for row in sweep if row.get("success")])
        if success_df.empty:
            continue
        plotted = True
        label = summary["experiment_name"]
        ax.plot(
            success_df["batch_size"],
            success_df["peak_memory_gb"],
            marker="o",
            linewidth=2.5,
            label=label,
        )
        last = success_df.iloc[-1]
        ax.scatter(
            [last["batch_size"]],
            [last["peak_memory_gb"]],
            marker="*",
            s=260,
            zorder=5,
        )
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Budgeted Scaling: Peak Memory vs Batch Size")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak Memory (GB)")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_max_runnable_batch(micro_dir: Path, output_path: Path) -> None:
    frame = _collect_benchmark_summaries(micro_dir)
    if frame.empty:
        return
    if "benchmark_mode" not in frame.columns:
        frame["benchmark_mode"] = "max_batch"
    frame = frame[frame["benchmark_mode"] == "max_batch"].copy()
    frame = frame.dropna(subset=["max_batch_size"])
    if frame.empty:
        return
    frame = frame.sort_values("max_batch_size", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(frame, x="label", y="max_batch_size", ax=ax, color="#edae49")
    budget = _shared_non_null_value(frame, "memory_budget_gb")
    if budget is not None:
        ax.set_title(f"Max Runnable Batch under Shared Memory Budget ({float(budget):.2f} GB)")
    else:
        ax.set_title("Max Runnable Batch by Optimizer")
    ax.set_xlabel("")
    ax.set_ylabel("Max Runnable Batch")
    ax.tick_params(axis="x", rotation=20)
    for idx, row in frame.reset_index(drop=True).iterrows():
        peak_memory = row["peak_memory_gb"]
        if peak_memory is not None:
            ax.text(idx, row["max_batch_size"] + 0.3, f"{peak_memory:.1f} GB", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_step_spikes(micro_dir: Path, output_path: Path) -> None:
    trace_paths = sorted(micro_dir.glob("*/step_times.jsonl"))
    available_columns = []
    frames: list[tuple[str, pd.DataFrame]] = []
    for trace_path in trace_paths:
        frame = _read_jsonl(trace_path)
        if frame.empty:
            continue
        frames.append((_canonical_experiment_name(trace_path.parent.name), frame))
        for column in ("forward_time_s", "backward_time_s", "optimizer_step_time_s"):
            if column in frame.columns and column not in available_columns:
                available_columns.append(column)
    if not frames or not available_columns:
        return

    titles = {
        "forward_time_s": "Forward Time",
        "backward_time_s": "Backward Time",
        "optimizer_step_time_s": "Optimizer Step Time",
    }
    fig, axes = plt.subplots(len(available_columns), 1, figsize=(11, 3.2 * len(available_columns) + 1), sharex=True)
    if len(available_columns) == 1:
        axes = [axes]
    plotted = False
    for label, frame in frames:
        plotted = True
        for ax, column in zip(axes, available_columns):
            if column in frame.columns:
                ax.plot(frame["step"], frame[column], label=label)
    if not plotted:
        plt.close(fig)
        return
    for ax, column in zip(axes, available_columns):
        ax.set_title(titles[column])
        ax.set_ylabel("Seconds")
    axes[-1].set_xlabel("Step")
    axes[0].legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_table7_like(micro_dir: Path, output_csv: Path) -> pd.DataFrame:
    frame = _collect_benchmark_summaries(micro_dir)
    if frame.empty:
        frame.to_csv(output_csv, index=False)
        return frame
    columns = [
        "label",
        "benchmark_mode",
        "measured_batch_size",
        "memory_budget_gb",
        "avg_forward_time_s",
        "avg_backward_time_s",
        "avg_optimizer_step_time_s",
        "avg_total_step_time_s",
        "tokens_per_second",
        "max_batch_size",
    ]
    available = [column for column in columns if column in frame.columns]
    frame = frame[available].rename(columns={"label": "experiment_name"})
    frame = frame.sort_values("experiment_name")
    frame.to_csv(output_csv, index=False)
    return frame


def plot_step_time_breakdown(micro_dir: Path, output_path: Path) -> None:
    frame = _collect_benchmark_summaries(micro_dir)
    if frame.empty:
        return
    required = ["avg_forward_time_s", "avg_backward_time_s", "avg_optimizer_step_time_s"]
    if any(column not in frame.columns for column in required):
        return
    frame = frame.set_index("label")[required]
    frame = frame.rename(
        columns={
            "avg_forward_time_s": "forward",
            "avg_backward_time_s": "backward",
            "avg_optimizer_step_time_s": "optimizer_step",
        }
    )
    ax = frame.plot(kind="bar", stacked=True, figsize=(11, 6), color=["#2a9d8f", "#e9c46a", "#d1495b"])
    summary_frame = _collect_benchmark_summaries(micro_dir)
    mode = _shared_non_null_value(summary_frame, "benchmark_mode")
    fixed_batch = _shared_non_null_value(summary_frame, "measured_batch_size")
    budget = _shared_non_null_value(summary_frame, "memory_budget_gb")
    if mode == "fixed_batch" and fixed_batch is not None:
        ax.set_title(f"Average Step-Time Breakdown at Fixed Batch (batch={int(fixed_batch)})")
    elif budget is not None:
        ax.set_title(f"Average Step-Time Breakdown under Shared Memory Budget ({float(budget):.2f} GB)")
    else:
        ax.set_title("Average Step-Time Breakdown")
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_sharpness(full_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for sharp_path in sorted(full_dir.glob("*/sharpness_analysis.jsonl")):
        frame = _read_jsonl(sharp_path)
        if frame.empty:
            continue
        plotted = True
        ax.plot(frame["step"], frame["sharpness"], marker="o", label=sharp_path.parent.name)
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Directional Sharpness")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$v^T H v$")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ablation_curves(ablation_dir: Path, output_path: Path, names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in names:
        metrics_path = ablation_dir / name / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        frame = _read_jsonl(metrics_path)
        train_frame = frame.dropna(subset=["train_loss"])
        ax.plot(train_frame["step"], train_frame["train_loss"], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_all_plots(results_root: str | Path) -> None:
    root = Path(results_root)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    full_dir = root / "full_pretrain"
    finetune_dir = root / "finetune"
    micro_dir = root / "microbench"
    ablation_dir = root / "ablation"

    if full_dir.exists():
        plot_convergence(full_dir, figure_dir / "convergence_race.png")
        plot_validation_convergence(full_dir, figure_dir / "validation_convergence.png")
        table_df = make_table2_like(full_dir, figure_dir / "table2_like.csv")
        if not table_df.empty:
            pareto_df, pareto_xlabel = _merge_fixed_batch_memory(table_df, micro_dir if micro_dir.exists() else None)
            plot_pareto(pareto_df, figure_dir / "pareto_frontier.png", x_label=pareto_xlabel)
        plot_scaling_ratio(full_dir, figure_dir / "scaling_ratio.png")
        plot_figure4_scaling(full_dir, figure_dir / "figure4_scaling_ratio.png")
        plot_sharpness(full_dir, figure_dir / "sharpness_curve.png")

    if finetune_dir.exists():
        plot_convergence(finetune_dir, figure_dir / "finetune_convergence.png")
        plot_validation_convergence(finetune_dir, figure_dir / "finetune_validation_loss.png")
        plot_finetune_accuracy(finetune_dir, figure_dir / "finetune_validation_accuracy.png")
        make_finetune_table(finetune_dir, figure_dir / "finetune_summary.csv")

    if micro_dir.exists():
        plot_memory_breakdown(micro_dir, figure_dir / "memory_breakdown.png")
        plot_batch_throughput(micro_dir, figure_dir / "batch_throughput.png")
        plot_budgeted_scaling_curves(micro_dir, figure_dir / "budgeted_scaling_curves.png")
        plot_max_runnable_batch(micro_dir, figure_dir / "max_runnable_batch.png")
        plot_step_time_breakdown(micro_dir, figure_dir / "step_time_breakdown.png")
        plot_step_spikes(micro_dir, figure_dir / "optimizer_step_spikes.png")
        make_table7_like(micro_dir, figure_dir / "table7_like.csv")

    if ablation_dir.exists():
        plot_ablation_curves(
            ablation_dir,
            figure_dir / "rank_ablation.png",
            [
                "galore_rank_32",
                "galore_rank_8",
                "galore_rank_1",
                "apollo_rank_32",
                "apollo_rank_8",
                "apollo_mini_rank_1",
            ],
        )
        plot_ablation_curves(
            ablation_dir,
            figure_dir / "granularity_r1.png",
            ["apollo_channel_rank_1", "apollo_mini_rank_1"],
        )
