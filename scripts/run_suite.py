from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apollo_story.benchmark import benchmark_experiment
from apollo_story.config import deep_update, load_yaml
from apollo_story.finetune import finetune_experiment
from apollo_story.train import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="Path to suite yaml.")
    parser.add_argument("--output-root", default="results", help="Output root.")
    parser.add_argument(
        "--resume",
        default="none",
        help="Resume mode for training runs: none, latest, or an explicit checkpoint path.",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite).resolve()
    suite = load_yaml(suite_path)
    shared_ref = Path(suite["shared_config"])
    if not shared_ref.is_absolute():
        suite_relative = (suite_path.parent / shared_ref).resolve()
        project_relative = (PROJECT_ROOT / shared_ref).resolve()
        shared_ref = suite_relative if suite_relative.exists() else project_relative
    shared = load_yaml(shared_ref)
    suite_root = Path(args.output_root) / suite["suite_name"]

    for experiment in suite["experiments"]:
        config = deep_update(shared, experiment.get("overrides", {}))
        config["experiment_name"] = experiment["name"]
        config["output_root"] = str(suite_root)
        run_kind = config.get("run_kind", "train")
        if run_kind == "benchmark":
            benchmark_experiment(config)
        elif run_kind == "finetune_mc":
            finetune_experiment(config, resume=args.resume)
        else:
            train_experiment(config, resume=args.resume)


if __name__ == "__main__":
    main()
