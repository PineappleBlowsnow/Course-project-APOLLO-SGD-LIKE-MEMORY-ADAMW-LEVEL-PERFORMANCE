from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


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
    frame = pd.DataFrame(rows).sort_values("experiment_name")
    frame.to_csv(output_csv, index=False)
    return frame


def plot_pareto(table_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(table_df["peak_memory_gb"], table_df["final_eval_loss"], s=180, c="#d1495b")
    for _, row in table_df.iterrows():
        ax.annotate(
            row["experiment_name"],
            (row["peak_memory_gb"], row["final_eval_loss"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Pareto Frontier: Peak Memory vs Final Validation Loss")
    ax.set_xlabel("Peak Memory (GB)")
    ax.set_ylabel("Final Validation Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_scaling_ratio(full_dir: Path, output_path: Path) -> None:
    adamw_path = full_dir / "adamw" / "scaling_traces.jsonl"
    apollo_14_path = full_dir / "apollo_rank_1_4" / "scaling_traces.jsonl"
    apollo_18_path = full_dir / "apollo_rank_1_8" / "scaling_traces.jsonl"
    if not (adamw_path.exists() and apollo_14_path.exists() and apollo_18_path.exists()):
        return

    adamw = _read_jsonl(adamw_path)
    apollo_14 = _read_jsonl(apollo_14_path)
    apollo_18 = _read_jsonl(apollo_18_path)
    rows = []
    for label, frame in [("APOLLO n/4", apollo_14), ("APOLLO n/8", apollo_18)]:
        for _, row in frame.iterrows():
            matched = adamw[
                (adamw["step"] == row["step"])
                & (adamw["param_name"] == row["param_name"])
            ]
            if matched.empty:
                continue
            base = matched.iloc[0]["values"]
            target = row["values"]
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
    ax.set_title("Scaling Factor Ratio vs AdamW")
    ax.set_ylabel("APOLLO scale / AdamW scale")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_memory_breakdown(micro_dir: Path, output_path: Path) -> None:
    rows = []
    for summary_path in sorted(micro_dir.glob("*/benchmark_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        breakdown = summary.get("memory_breakdown")
        if not breakdown:
            continue
        breakdown["experiment_name"] = summary["experiment_name"]
        rows.append(breakdown)
    if not rows:
        return
    frame = pd.DataFrame(rows).set_index("experiment_name")
    components = ["weights_gb", "activations_gb", "gradients_gb", "optimizer_states_gb"]
    ax = frame[components].plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_title("Memory Breakdown")
    ax.set_xlabel("")
    ax.set_ylabel("GB")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_batch_throughput(micro_dir: Path, output_path: Path) -> None:
    rows = []
    for summary_path in sorted(micro_dir.glob("*/benchmark_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values("max_batch_size", ascending=False)
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.bar(frame["experiment_name"], frame["max_batch_size"], color="#edae49", alpha=0.8)
    ax2.plot(frame["experiment_name"], frame["tokens_per_second"], color="#00798c", marker="o", linewidth=3)
    ax1.set_ylabel("Max Batch Size")
    ax2.set_ylabel("Tokens / second")
    ax1.set_title("Max Batch Size and Throughput Scaling")
    ax1.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_step_spikes(micro_dir: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for trace_path in sorted(micro_dir.glob("*/step_times.jsonl")):
        frame = _read_jsonl(trace_path)
        label = trace_path.parent.name
        axes[0].plot(frame["step"], frame["backward_time_s"], label=label)
        axes[1].plot(frame["step"], frame["optimizer_step_time_s"], label=label)
    axes[0].set_title("Backward Time")
    axes[0].set_ylabel("Seconds")
    axes[1].set_title("Optimizer Step Time")
    axes[1].set_ylabel("Seconds")
    axes[1].set_xlabel("Step")
    axes[0].legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_table7_like(micro_dir: Path, output_csv: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(micro_dir.glob("*/benchmark_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "experiment_name": summary["experiment_name"],
                "avg_backward_time_s": summary["avg_backward_time_s"],
                "avg_optimizer_step_time_s": summary["avg_optimizer_step_time_s"],
                "tokens_per_second": summary["tokens_per_second"],
                "max_batch_size": summary["max_batch_size"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("experiment_name")
    frame.to_csv(output_csv, index=False)
    return frame


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
    micro_dir = root / "microbench"
    ablation_dir = root / "ablation"

    if full_dir.exists():
        plot_convergence(full_dir, figure_dir / "convergence_race.png")
        table_df = make_table2_like(full_dir, figure_dir / "table2_like.csv")
        if not table_df.empty:
            plot_pareto(table_df, figure_dir / "pareto_frontier.png")
        plot_scaling_ratio(full_dir, figure_dir / "scaling_ratio.png")
        plot_sharpness(full_dir, figure_dir / "sharpness_curve.png")

    if micro_dir.exists():
        plot_memory_breakdown(micro_dir, figure_dir / "memory_breakdown.png")
        plot_batch_throughput(micro_dir, figure_dir / "batch_throughput.png")
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

