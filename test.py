from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from humanoid_rl import ASSIGNMENT_ENV_ID
from humanoid_rl.artifacts import checkpoint_artifacts
from humanoid_rl.config import load_config
from humanoid_rl.evaluation import load_vecnormalize_for_inference, run_raw_reward_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained Humanoid-v5 policy. The seed argument is passed to "
            "env.reset(seed=...) for assignment evaluation compatibility."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-mode", choices=["none", "human"], default="none")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.run_dir / "config.json")
    if args.checkpoint_step is not None:
        if args.model_path is not None or args.vecnormalize_path is not None:
            raise ValueError(
                "--checkpoint-step cannot be combined with --model-path or "
                "--vecnormalize-path."
            )
        model_path, vecnormalize_path = checkpoint_artifacts(args.run_dir, args.checkpoint_step)
    else:
        model_path = args.model_path or args.run_dir / "models" / "latest_model.zip"
        vecnormalize_path = args.vecnormalize_path or args.run_dir / "models" / "vecnormalize_latest.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if bool(config["normalize_observation"]) and not vecnormalize_path.exists():
        raise FileNotFoundError(
            "VecNormalize statistics are required for normalized-observation inference: "
            f"{vecnormalize_path}"
        )

    model = PPO.load(str(model_path), device=args.device)
    normalizer = load_vecnormalize_for_inference(vecnormalize_path if vecnormalize_path.exists() else None)
    render_mode = None if args.render_mode == "none" else args.render_mode
    env_id = config.get("env_id", ASSIGNMENT_ENV_ID)

    results = run_raw_reward_episodes(
        model=model,
        env_id=env_id,
        normalizer=normalizer,
        seed=args.seed,
        episodes=args.episodes,
        deterministic=bool(args.deterministic),
        render_mode=render_mode,
    )

    for result in results:
        print(
            f"seed={result.seed} episode={result.episode_index} "
            f"raw_reward={result.reward:.3f} length={result.length} "
            f"terminated={result.terminated} truncated={result.truncated}"
        )

    mean_reward = sum(r.reward for r in results) / len(results)
    print(f"mean_raw_reward={mean_reward:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
