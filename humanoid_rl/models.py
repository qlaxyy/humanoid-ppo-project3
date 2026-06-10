from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def algorithm_name(config: dict[str, Any]) -> str:
    if config.get("algorithm"):
        name = str(config["algorithm"]).lower()
    elif "sac" in config:
        name = "sac"
    elif "td3" in config:
        name = "td3"
    else:
        name = "ppo"

    if name not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm={name!r}. Choose from {sorted(ALGORITHMS)}.")
    return name


def load_model_for_config(
    model_path: Path,
    config: dict[str, Any],
    device: str = "auto",
    env: Any | None = None,
    seed: int | None = None,
) -> BaseAlgorithm:
    algorithm = ALGORITHMS[algorithm_name(config)]
    kwargs: dict[str, Any] = {"device": device}
    if env is not None:
        kwargs["env"] = env
    if seed is not None:
        kwargs["seed"] = seed
    return algorithm.load(str(model_path), **kwargs)
