from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apollo_story.config import load_yaml
from apollo_story.train import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--resume",
        default="none",
        help="Resume mode: none, latest, or an explicit checkpoint path.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    train_experiment(config, resume=args.resume)


if __name__ == "__main__":
    main()
