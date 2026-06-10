from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize


def make_single_env(
    env_id: str,
    seed: int,
    rank: int = 0,
    monitor_dir: Path | None = None,
    render_mode: str | None = None,
):
    """Create one unmodified Gymnasium environment for vectorized training."""

    def _init():
        actual_seed = seed + rank
        env = gym.make(env_id, render_mode=render_mode)
        env.action_space.seed(actual_seed)
        env.observation_space.seed(actual_seed)

        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(monitor_dir / f"env_{rank}"))
        else:
            env = Monitor(env)

        env.reset(seed=actual_seed)
        return env

    return _init


def make_vector_env(
    env_id: str,
    seed: int,
    n_envs: int,
    vec_env_type: str,
    monitor_dir: Path | None,
    normalize_observation: bool,
    normalize_reward: bool,
    norm_gamma: float,
    clip_obs: float,
    vecnormalize_path: Path | None = None,
    training: bool = True,
) -> VecEnv:
    env_fns = [
        make_single_env(env_id, seed=seed, rank=rank, monitor_dir=monitor_dir)
        for rank in range(n_envs)
    ]

    if vec_env_type == "subproc" and n_envs > 1:
        env: VecEnv = SubprocVecEnv(env_fns, start_method="spawn")
    elif vec_env_type == "dummy":
        env = DummyVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    env.seed(seed)

    if vecnormalize_path is not None:
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = training
        env.norm_reward = normalize_reward if training else False
        return env

    if normalize_observation or normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=normalize_observation,
            norm_reward=normalize_reward,
            gamma=norm_gamma,
            clip_obs=clip_obs,
        )
        env.training = training
        if not training:
            env.norm_reward = False

    return env

