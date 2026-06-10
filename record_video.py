from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from humanoid_rl import ASSIGNMENT_ENV_ID
from humanoid_rl.artifacts import checkpoint_artifacts
from humanoid_rl.config import load_config
from humanoid_rl.evaluation import (
    EpisodeResult,
    load_vecnormalize_for_inference,
    normalize_observation,
)
from humanoid_rl.io import timestamp_for_path, write_json
from humanoid_rl.models import load_model_for_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record an mp4 video of a trained Humanoid-v5 policy."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name-prefix", type=str, default=None)
    return parser.parse_args()


def resolve_artifacts(args: argparse.Namespace, config: dict) -> tuple[Path, Path | None]:
    if args.checkpoint_step is not None:
        if args.model_path is not None or args.vecnormalize_path is not None:
            raise ValueError(
                "--checkpoint-step cannot be combined with --model-path or "
                "--vecnormalize-path."
            )
        return checkpoint_artifacts(
            args.run_dir,
            args.checkpoint_step,
            require_vecnormalize=bool(config["normalize_observation"]),
        )

    model_path = args.model_path or args.run_dir / "models" / "latest_model.zip"
    vecnormalize_path = args.vecnormalize_path or args.run_dir / "models" / "vecnormalize_latest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnormalize_path.exists():
        vecnormalize_path = None
    return model_path, vecnormalize_path


def main() -> int:
    args = parse_args()
    config = load_config(args.run_dir / "config.json")
    env_id = config.get("env_id", ASSIGNMENT_ENV_ID)
    model_path, vecnormalize_path = resolve_artifacts(args, config)

    if bool(config["normalize_observation"]) and vecnormalize_path is None:
        raise FileNotFoundError("VecNormalize statistics are required for this model.")

    video_dir = args.video_dir or args.run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_for_path()
    prefix = args.name_prefix or f"{args.run_dir.name}_seed{args.seed}_{stamp}"

    model = load_model_for_config(model_path, config, device=args.device)
    normalizer = load_vecnormalize_for_inference(vecnormalize_path)

    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: True,
        name_prefix=prefix,
    )

    results: list[EpisodeResult] = []
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            obs, _ = env.reset(seed=episode_seed)
            total_reward = 0.0
            length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                model_obs = normalize_observation(normalizer, obs)
                action, _ = model.predict(model_obs, deterministic=bool(args.deterministic))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1

            results.append(
                EpisodeResult(
                    seed=episode_seed,
                    episode_index=episode_index,
                    reward=total_reward,
                    length=length,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
    finally:
        env.close()

    mp4_files = sorted(video_dir.glob(f"{prefix}*.mp4"))
    payload = {
        "run_dir": str(args.run_dir),
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path else None,
        "checkpoint_step": args.checkpoint_step,
        "seed": args.seed,
        "episodes": args.episodes,
        "deterministic": bool(args.deterministic),
        "results": [asdict(result) for result in results],
        "videos": [str(path) for path in mp4_files],
    }
    summary_path = video_dir / f"{prefix}_summary.json"
    write_json(summary_path, payload)

    for result in results:
        print(
            f"seed={result.seed} episode={result.episode_index} "
            f"raw_reward={result.reward:.3f} length={result.length} "
            f"terminated={result.terminated} truncated={result.truncated}"
        )
    print(f"Saved summary: {summary_path}")
    for path in mp4_files:
        print(f"Saved video:   {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
