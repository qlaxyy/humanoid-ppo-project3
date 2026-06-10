from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class EpisodeResult:
    seed: int
    episode_index: int
    reward: float
    length: int
    terminated: bool
    truncated: bool


def load_vecnormalize_for_inference(path: Path | None):
    if path is None:
        return None
    with path.open("rb") as f:
        normalizer = pickle.load(f)
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def normalize_observation(normalizer: Any, obs: np.ndarray) -> np.ndarray:
    if normalizer is None or not getattr(normalizer, "norm_obs", False):
        return np.asarray(obs, dtype=np.float32)

    obs_rms = normalizer.obs_rms
    normalized = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + normalizer.epsilon)
    normalized = np.clip(normalized, -normalizer.clip_obs, normalizer.clip_obs)
    return np.asarray(normalized, dtype=np.float32)


def run_raw_reward_episodes(
    model: BaseAlgorithm,
    env_id: str,
    normalizer: Any,
    seed: int,
    episodes: int,
    deterministic: bool = True,
    render_mode: str | None = None,
) -> list[EpisodeResult]:
    """Evaluate with env.reset(seed=...) and raw Gymnasium rewards."""
    results: list[EpisodeResult] = []
    env = gym.make(env_id, render_mode=render_mode)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            obs, _ = env.reset(seed=episode_seed)
            total_reward = 0.0
            length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                model_obs = normalize_observation(normalizer, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1

                if render_mode == "human":
                    env.render()

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

    return results


def summarize_results(results: list[EpisodeResult]) -> dict[str, float | int]:
    rewards = np.asarray([r.reward for r in results], dtype=np.float64)
    lengths = np.asarray([r.length for r in results], dtype=np.float64)
    return {
        "episodes": int(len(results)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
    }
