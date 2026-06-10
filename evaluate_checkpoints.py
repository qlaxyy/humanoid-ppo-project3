from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from humanoid_rl import ASSIGNMENT_ENV_ID
from humanoid_rl.config import load_config
from humanoid_rl.evaluation import (
    load_vecnormalize_for_inference,
    run_raw_reward_episodes,
    summarize_results,
)
from humanoid_rl.io import timestamp_for_path
from humanoid_rl.models import load_model_for_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved PPO checkpoints with raw Humanoid-v5 rewards."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--every", type=int, default=500_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_model_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else -1


def find_vecnormalize(models_dir: Path, step: int) -> Path | None:
    candidates = [
        models_dir / f"checkpoint_model_vecnormalize_{step}_steps.pkl",
        models_dir / f"vecnormalize_{step}_steps.pkl",
    ]
    return next((path for path in candidates if path.exists()), None)


def select_checkpoints(run_dir: Path, requested_steps: list[int] | None, every: int) -> list[Path]:
    models_dir = run_dir / "models"
    checkpoints = sorted(
        models_dir.glob("checkpoint_model_*_steps.zip"),
        key=checkpoint_step,
    )
    if requested_steps:
        wanted = set(requested_steps)
        checkpoints = [path for path in checkpoints if checkpoint_step(path) in wanted]
    elif every > 0:
        checkpoints = [path for path in checkpoints if checkpoint_step(path) % every == 0]

    if not checkpoints:
        raise FileNotFoundError(
            f"No matching checkpoint_model_*_steps.zip files found under {models_dir}."
        )
    return checkpoints


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = ["step", "mean_reward", "std_reward", "min_reward", "max_reward", "mean_length"]
    widths = {
        column: max(len(column), *(len(format_value(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(format_value(row[column]).ljust(widths[column]) for column in columns))


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    config = load_config(run_dir / "config.json")
    env_id = config.get("env_id", ASSIGNMENT_ENV_ID)
    models_dir = run_dir / "models"
    checkpoints = select_checkpoints(run_dir, args.steps, args.every)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        step = checkpoint_step(checkpoint_path)
        vecnormalize_path = find_vecnormalize(models_dir, step)
        if bool(config["normalize_observation"]) and vecnormalize_path is None:
            raise FileNotFoundError(
                f"Missing VecNormalize statistics for checkpoint step {step}."
            )

        print(f"\nEvaluating step {step}: {checkpoint_path.name}")
        model = load_model_for_config(checkpoint_path, config, device=args.device)
        normalizer = load_vecnormalize_for_inference(vecnormalize_path)

        episode_results = []
        for seed in args.seeds:
            episode_results.extend(
                run_raw_reward_episodes(
                    model=model,
                    env_id=env_id,
                    normalizer=normalizer,
                    seed=seed,
                    episodes=args.episodes_per_seed,
                    deterministic=bool(args.deterministic),
                )
            )

        summary = summarize_results(episode_results)
        row = {
            "step": step,
            **summary,
            "model_path": str(checkpoint_path),
            "vecnormalize_path": str(vecnormalize_path),
        }
        rows.append(row)
        details.append(
            {
                "step": step,
                "summary": summary,
                "episodes": [asdict(result) for result in episode_results],
                "model_path": str(checkpoint_path),
                "vecnormalize_path": str(vecnormalize_path),
            }
        )
        print(
            f"step={step} mean_reward={summary['mean_reward']:.3f} "
            f"mean_length={summary['mean_length']:.1f}"
        )

    rows.sort(key=lambda row: float(row["mean_reward"]), reverse=True)
    print("\nCheckpoint sweep summary")
    print_table(rows)

    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_for_path()
    csv_path = eval_dir / f"checkpoint_sweep_{stamp}.csv"
    json_path = eval_dir / f"checkpoint_sweep_{stamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step",
            "episodes",
            "mean_reward",
            "std_reward",
            "min_reward",
            "max_reward",
            "mean_length",
            "model_path",
            "vecnormalize_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "seeds": args.seeds,
                "episodes_per_seed": args.episodes_per_seed,
                "rows_sorted_by_mean_reward": rows,
                "details": details,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    best = rows[0]
    print("\nBest checkpoint")
    print(f"  step:              {best['step']}")
    print(f"  mean_reward:       {best['mean_reward']:.3f}")
    print(f"  model_path:        {best['model_path']}")
    print(f"  vecnormalize_path: {best['vecnormalize_path']}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
