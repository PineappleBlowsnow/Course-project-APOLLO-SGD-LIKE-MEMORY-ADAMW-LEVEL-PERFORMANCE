from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    figure_dir = Path(args.results_root) / "figures"
    target_dir = Path("poster") / "assets" / "generated"
    target_dir.mkdir(parents=True, exist_ok=True)

    for path in figure_dir.glob("*"):
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)


if __name__ == "__main__":
    main()

