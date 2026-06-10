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
        description="Evaluate a trained Humanoid-v5 model with raw environment rewards."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-vecnormalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    config = load_config(run_dir / "config.json")

    if args.checkpoint_step is not None:
        if args.model_path is not None or args.vecnormalize_path is not None:
            raise ValueError(
                "--checkpoint-step cannot be combined with --model-path or "
                "--vecnormalize-path."
            )
        model_path, vecnormalize_path = checkpoint_artifacts(
            run_dir,
            args.checkpoint_step,
            require_vecnormalize=bool(config["normalize_observation"]),
        )
    else:
        model_path = args.model_path or run_dir / "models" / "latest_model.zip"
        vecnormalize_path = args.vecnormalize_path or run_dir / "models" / "vecnormalize_latest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if bool(config["normalize_observation"]) and (
        vecnormalize_path is None or not vecnormalize_path.exists()
    ):
        if not args.allow_missing_vecnormalize:
            raise FileNotFoundError(
                "VecNormalize statistics are required because training normalized "
                f"observations, but this file is missing: {vecnormalize_path}"
            )
        vecnormalize_path = None

    model = load_model_for_config(model_path, config, device=args.device)
    normalizer = load_vecnormalize_for_inference(vecnormalize_path)

    all_results = []
    env_id = config.get("env_id", ASSIGNMENT_ENV_ID)
    for seed in args.seeds:
        all_results.extend(
            run_raw_reward_episodes(
                model=model,
                env_id=env_id,
                normalizer=normalizer,
                seed=seed,
                episodes=args.episodes_per_seed,
                deterministic=bool(args.deterministic),
            )
        )

    summary = summarize_results(all_results)
    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_for_path()
    csv_path = eval_dir / f"raw_eval_{stamp}.csv"
    json_path = eval_dir / f"raw_eval_{stamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
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
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "deterministic": bool(args.deterministic),
        "summary": summary,
        "rewards": [float(r.reward) for r in all_results],
    }
    write_json(json_path, payload)

    print("Raw reward evaluation summary")
    print(f"  episodes:    {summary['episodes']}")
    print(f"  mean_reward: {summary['mean_reward']:.3f}")
    print(f"  std_reward:  {summary['std_reward']:.3f}")
    print(f"  min_reward:  {summary['min_reward']:.3f}")
    print(f"  max_reward:  {summary['max_reward']:.3f}")
    print(f"  mean_length: {summary['mean_length']:.1f}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print("Rewards:", np.array2string(np.asarray(payload["rewards"]), precision=3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
