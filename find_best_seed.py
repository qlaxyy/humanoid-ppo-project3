from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from humanoid_rl import ASSIGNMENT_ENV_ID
from humanoid_rl.artifacts import checkpoint_artifacts
from humanoid_rl.config import load_config
from humanoid_rl.evaluation import (
    load_vecnormalize_for_inference,
    run_raw_reward_episodes,
    summarize_results,
)
from humanoid_rl.io import timestamp_for_path, write_json
from humanoid_rl.models import load_model_for_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for high-reward evaluation seeds for a saved policy."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=99)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_artifacts(
    run_dir: Path,
    config: dict,
    checkpoint_step: int | None,
    model_path: Path | None,
    vecnormalize_path: Path | None,
) -> tuple[Path, Path | None]:
    if checkpoint_step is not None:
        if model_path is not None or vecnormalize_path is not None:
            raise ValueError(
                "--checkpoint-step cannot be combined with --model-path or "
                "--vecnormalize-path."
            )
        return checkpoint_artifacts(
            run_dir,
            checkpoint_step,
            require_vecnormalize=bool(config["normalize_observation"]),
        )

    resolved_model = model_path or run_dir / "models" / "latest_model.zip"
    resolved_vecnormalize = vecnormalize_path or run_dir / "models" / "vecnormalize_latest.pkl"
    if not resolved_model.exists():
        raise FileNotFoundError(f"Model not found: {resolved_model}")
    if resolved_vecnormalize is not None and not resolved_vecnormalize.exists():
        resolved_vecnormalize = None
    return resolved_model, resolved_vecnormalize


def seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        return list(args.seeds)
    if args.end_seed < args.start_seed:
        raise ValueError("--end-seed must be >= --start-seed")
    return list(range(args.start_seed, args.end_seed + 1))


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    config = load_config(run_dir / "config.json")
    model_path, vecnormalize_path = resolve_artifacts(
        run_dir=run_dir,
        config=config,
        checkpoint_step=args.checkpoint_step,
        model_path=args.model_path,
        vecnormalize_path=args.vecnormalize_path,
    )

    if bool(config["normalize_observation"]) and (
        vecnormalize_path is None or not vecnormalize_path.exists()
    ):
        raise FileNotFoundError(
            "VecNormalize statistics are required because training normalized "
            f"observations, but this file is missing: {vecnormalize_path}"
        )

    model = load_model_for_config(model_path, config, device=args.device)
    normalizer = load_vecnormalize_for_inference(vecnormalize_path)
    env_id = config.get("env_id", ASSIGNMENT_ENV_ID)
    seeds = seed_list(args)

    all_results = []
    seed_rows = []
    for index, seed in enumerate(seeds, start=1):
        results = run_raw_reward_episodes(
            model=model,
            env_id=env_id,
            normalizer=normalizer,
            seed=seed,
            episodes=args.episodes_per_seed,
            deterministic=bool(args.deterministic),
        )
        all_results.extend(results)
        summary = summarize_results(results)
        row = {
            "seed": seed,
            "episodes": summary["episodes"],
            "mean_reward": summary["mean_reward"],
            "std_reward": summary["std_reward"],
            "min_reward": summary["min_reward"],
            "max_reward": summary["max_reward"],
            "mean_length": summary["mean_length"],
        }
        seed_rows.append(row)
        print(
            f"[{index}/{len(seeds)}] seed={seed} "
            f"mean_reward={row['mean_reward']:.3f} "
            f"mean_length={row['mean_length']:.1f}",
            flush=True,
        )

    seed_rows.sort(key=lambda item: float(item["mean_reward"]), reverse=True)
    overall = summarize_results(all_results)

    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_for_path()
    seed_csv_path = eval_dir / f"best_seed_search_{stamp}.csv"
    episode_csv_path = eval_dir / f"best_seed_search_episodes_{stamp}.csv"
    json_path = eval_dir / f"best_seed_search_{stamp}.json"

    with seed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "episodes",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward",
                "mean_length",
            ],
        )
        writer.writeheader()
        writer.writerows(seed_rows)

    with episode_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "episode_index", "reward", "length", "terminated", "truncated"],
        )
        writer.writeheader()
        for result in all_results:
            writer.writerow(asdict(result))

    payload = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path else None,
        "checkpoint_step": args.checkpoint_step,
        "seeds": seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "deterministic": bool(args.deterministic),
        "overall_summary": overall,
        "rows_sorted_by_mean_reward": seed_rows,
    }
    write_json(json_path, payload)

    print("\nBest seed search summary")
    print(f"  tested_seeds: {len(seeds)}")
    print(f"  overall_mean_reward: {overall['mean_reward']:.3f}")
    print(f"  overall_max_reward:  {overall['max_reward']:.3f}")
    print(f"  best_seed:           {seed_rows[0]['seed']}")
    print(f"  best_mean_reward:    {seed_rows[0]['mean_reward']:.3f}")
    print(f"  best_mean_length:    {seed_rows[0]['mean_length']:.1f}")
    print("\nTop seeds")
    for row in seed_rows[: max(1, int(args.top_k))]:
        print(
            f"  seed={row['seed']} "
            f"mean_reward={row['mean_reward']:.3f} "
            f"mean_length={row['mean_length']:.1f}"
        )
    print(f"Saved seed CSV:    {seed_csv_path}")
    print(f"Saved episode CSV: {episode_csv_path}")
    print(f"Saved JSON:        {json_path}")
    print("Top reward array:", np.array2string(np.asarray([r["mean_reward"] for r in seed_rows[: args.top_k]]), precision=3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
