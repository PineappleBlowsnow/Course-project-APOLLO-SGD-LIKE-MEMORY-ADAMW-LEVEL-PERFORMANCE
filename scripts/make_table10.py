from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apollo_story.plotting import _read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    rows: list[pd.DataFrame] = []
    for exp_dir in sorted(suite_dir.iterdir()):
        sharpness_path = exp_dir / "sharpness_analysis.jsonl"
        if not exp_dir.is_dir() or not sharpness_path.exists():
            continue
        frame = _read_jsonl(sharpness_path)
        if frame.empty:
            continue
        frame = frame[["step", "sharpness"]].rename(columns={"sharpness": exp_dir.name})
        rows.append(frame)

    if not rows:
        raise RuntimeError(f"No sharpness_analysis.jsonl files found in {suite_dir}")

    merged = rows[0]
    for frame in rows[1:]:
        merged = merged.merge(frame, on="step", how="outer")
    merged = merged.sort_values("step")

    output_csv = Path(args.output_csv) if args.output_csv else suite_dir.parent / "figures" / "table10_like.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
