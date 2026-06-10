from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize raw reward evaluation results under runs/."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--write-csv", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_raw_eval(run_dir: Path) -> Path | None:
    eval_dir = run_dir / "evaluations"
    candidates = sorted(eval_dir.glob("raw_eval_*.json"))
    return candidates[-1] if candidates else None


def run_summary(run_dir: Path) -> dict[str, Any] | None:
    eval_path = latest_raw_eval(run_dir)
    if eval_path is None:
        return None

    payload = read_json(eval_path)
    metadata = read_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    ppo = config.get("ppo", {})
    summary = payload.get("summary", {})

    return {
        "run_id": run_dir.name,
        "steps": metadata.get("last_num_timesteps", ""),
        "mean_reward": summary.get("mean_reward", ""),
        "std_reward": summary.get("std_reward", ""),
        "min_reward": summary.get("min_reward", ""),
        "max_reward": summary.get("max_reward", ""),
        "mean_length": summary.get("mean_length", ""),
        "seed": config.get("seed", ""),
        "learning_rate": ppo.get("learning_rate", ""),
        "n_epochs": ppo.get("n_epochs", ""),
        "target_kl": ppo.get("target_kl", ""),
        "eval_file": str(eval_path),
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return ""
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_id",
        "steps",
        "mean_reward",
        "std_reward",
        "mean_length",
        "learning_rate",
        "n_epochs",
        "target_kl",
    ]
    widths = {
        column: max(len(column), *(len(format_value(row[column])) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(format_value(row[column]).ljust(widths[column]) for column in columns))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(format_value(row[column]) for column in columns) + "\n")


def main() -> int:
    args = parse_args()
    run_dirs = [path for path in sorted(args.runs_dir.iterdir()) if path.is_dir()]
    rows = [summary for path in run_dirs if (summary := run_summary(path)) is not None]
    rows.sort(key=lambda row: float(row["mean_reward"]), reverse=True)

    if not rows:
        print(f"No raw_eval_*.json files found under {args.runs_dir}.")
        return 0

    print_table(rows)
    if args.write_csv is not None:
        write_csv(args.write_csv, rows)
        print(f"\nSaved summary CSV: {args.write_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

