from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from humanoid_rl import ASSIGNMENT_ENV_ID, MAX_ENV_STEPS


DEFAULT_CONFIG: dict[str, Any] = {
    "env_id": ASSIGNMENT_ENV_ID,
    "seed": 3407,
    "target_steps": 1_000_000,
    "output_dir": "runs",
    "n_envs": 4,
    "vec_env": "subproc",
    "device": "auto",
    "torch_threads": 1,
    "normalize_observation": True,
    "normalize_reward": True,
    "norm_gamma": 0.99,
    "clip_obs": 10.0,
    "checkpoint_freq": 100_000,
    "eval_freq": 100_000,
    "eval_episodes": 3,
    "log_interval": 1,
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 1250,
        "batch_size": 250,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_net_arch": [256, 256],
        "value_net_arch": [256, 256],
    },
}


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into a copy of base."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return deep_update(config, loaded)


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def validate_config(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    if config["env_id"] != ASSIGNMENT_ENV_ID:
        raise ValueError(
            f"Assignment requires env_id={ASSIGNMENT_ENV_ID!r}; "
            f"got {config['env_id']!r}."
        )

    target_steps = int(config["target_steps"])
    if target_steps <= 0:
        raise ValueError("target_steps must be positive.")
    if target_steps > MAX_ENV_STEPS:
        raise ValueError(
            f"target_steps={target_steps} exceeds assignment limit {MAX_ENV_STEPS}."
        )

    n_envs = int(config["n_envs"])
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1.")

    ppo = config["ppo"]
    rollout_size = int(ppo["n_steps"]) * n_envs
    batch_size = int(ppo["batch_size"])
    if batch_size > rollout_size:
        raise ValueError(
            f"batch_size={batch_size} is larger than rollout_size={rollout_size}."
        )
    if rollout_size % batch_size != 0:
        warnings.append(
            "PPO rollout_size is not divisible by batch_size; "
            "Stable-Baselines3 will use a truncated final minibatch."
        )
    if target_steps % rollout_size != 0:
        raise ValueError(
            f"target_steps={target_steps} is not divisible by PPO rollout_size="
            f"n_steps*n_envs={rollout_size}. This could make PPO overshoot the "
            "assignment step limit. Adjust target_steps, n_steps, or n_envs."
        )

    if config.get("env_kwargs"):
        raise ValueError(
            "env_kwargs are disabled for this assignment to avoid changing "
            "MuJoCo physics or reward semantics."
        )

    return warnings
