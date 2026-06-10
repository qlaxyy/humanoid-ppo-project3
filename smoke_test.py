from __future__ import annotations

import argparse
import importlib.metadata

import gymnasium as gym

from humanoid_rl import ASSIGNMENT_ENV_ID


EXPECTED_VERSIONS = {
    "gymnasium": "1.2.3",
    "mujoco": "3.8.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check required versions and run a short random Humanoid-v5 rollout."
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--no-strict-versions", action="store_true")
    return parser.parse_args()


def check_versions(strict: bool) -> None:
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        print(f"{package}=={actual}")
        if strict and actual != expected:
            raise RuntimeError(
                f"{package} version mismatch: expected {expected}, got {actual}"
            )


def main() -> int:
    args = parse_args()
    check_versions(strict=not args.no_strict_versions)

    env = gym.make(ASSIGNMENT_ENV_ID)
    try:
        env.action_space.seed(args.seed)
        obs, info = env.reset(seed=args.seed)
        total_reward = 0.0
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        print(
            f"Smoke test finished: steps<={args.steps}, "
            f"raw_reward={total_reward:.3f}, obs_shape={getattr(obs, 'shape', None)}"
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

