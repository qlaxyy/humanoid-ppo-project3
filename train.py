from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from humanoid_rl import MAX_ENV_STEPS
from humanoid_rl.callbacks import (
    MetadataCallback,
    SaveVecNormalizeCallback,
    callback_freq,
)
from humanoid_rl.config import load_config, validate_config, write_config
from humanoid_rl.envs import make_vector_env
from humanoid_rl.io import read_json, runtime_info, timestamp_for_path, utc_now, write_json
from humanoid_rl.reproducibility import configure_torch_threads, seed_everything


DEFAULT_CONFIG_PATH = Path("configs/ppo_humanoid_colab.json")

ACTIVATION_FNS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on Gymnasium Humanoid-v5 with reproducible logging."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_schedule(value: float | str) -> float | Callable[[float], float]:
    if isinstance(value, str):
        if value.startswith("lin_"):
            initial_value = float(value.removeprefix("lin_"))

            def schedule(progress_remaining: float) -> float:
                return progress_remaining * initial_value

            return schedule
        return float(value)
    return float(value)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    direct_overrides = {
        "seed": args.seed,
        "target_steps": args.target_steps,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
    }
    for key, value in direct_overrides.items():
        if value is not None:
            config[key] = value

    ppo_overrides = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
    }
    for key, value in ppo_overrides.items():
        if value is not None:
            config["ppo"][key] = value

    return config


def load_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.resume_from is not None and args.config is None:
        config_path = args.resume_from / "config.json"
    else:
        config_path = args.config or DEFAULT_CONFIG_PATH

    config = load_config(config_path if config_path.exists() else None)
    return apply_overrides(config, args)


def create_or_resume_run_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.resume_from is not None:
        run_dir = args.resume_from
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        return run_dir

    output_dir = Path(config["output_dir"])
    config_label = (args.config.stem if args.config is not None else "ppo_humanoid")
    run_name = args.run_name or f"{timestamp_for_path()}_seed{config['seed']}_{config_label}"
    run_dir = output_dir / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else -1


def find_resume_artifacts(run_dir: Path) -> tuple[Path, Path | None]:
    models_dir = run_dir / "models"
    model_path = models_dir / "latest_model.zip"
    if model_path.exists():
        vecnormalize_path = models_dir / "vecnormalize_latest.pkl"
        return model_path, vecnormalize_path if vecnormalize_path.exists() else None

    checkpoints = sorted(
        models_dir.glob("checkpoint_model_*_steps.zip"),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(
            "No resume model found. Expected latest_model.zip or "
            f"checkpoint_model_*_steps.zip under {models_dir}."
        )

    model_path = checkpoints[-1]
    steps = checkpoint_step(model_path)
    vecnormalize_candidates = [
        models_dir / f"checkpoint_model_vecnormalize_{steps}_steps.pkl",
        models_dir / f"vecnormalize_{steps}_steps.pkl",
        models_dir / "vecnormalize_latest.pkl",
    ]
    vecnormalize_path = next(
        (path for path in vecnormalize_candidates if path.exists()),
        None,
    )
    print(f"Resuming from checkpoint: {model_path}")
    if vecnormalize_path is not None:
        print(f"Loading VecNormalize statistics: {vecnormalize_path}")
    else:
        print("[warning] No VecNormalize statistics found for resume.", file=sys.stderr)
    return model_path, vecnormalize_path


def build_model(config: dict[str, Any], train_env, tensorboard_dir: Path) -> PPO:
    ppo = config["ppo"]
    activation_name = str(ppo.get("activation_fn", "tanh")).lower()
    if activation_name not in ACTIVATION_FNS:
        raise ValueError(
            f"Unsupported activation_fn={activation_name!r}. "
            f"Choose from {sorted(ACTIVATION_FNS)}."
        )
    policy_kwargs = {
        "activation_fn": ACTIVATION_FNS[activation_name],
        "net_arch": {
            "pi": list(ppo["policy_net_arch"]),
            "vf": list(ppo["value_net_arch"]),
        },
        "ortho_init": bool(ppo.get("ortho_init", True)),
    }
    if ppo.get("log_std_init") is not None:
        policy_kwargs["log_std_init"] = float(ppo["log_std_init"])

    return PPO(
        policy=ppo["policy"],
        env=train_env,
        learning_rate=make_schedule(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=make_schedule(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        vf_coef=float(ppo["vf_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        target_kl=ppo.get("target_kl"),
        policy_kwargs=policy_kwargs,
        seed=int(config["seed"]),
        device=config["device"],
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
    )


def main() -> int:
    args = parse_args()
    config = load_effective_config(args)

    for warning in validate_config(config):
        print(f"[warning] {warning}", file=sys.stderr)

    seed = int(config["seed"])
    seed_everything(seed)
    configure_torch_threads(int(config["torch_threads"]))

    run_dir = create_or_resume_run_dir(config, args)
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    monitor_dir = run_dir / "monitor"
    eval_dir = run_dir / "evaluations"
    for directory in [models_dir, logs_dir, monitor_dir, eval_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    resume_model_path: Path | None = None
    resume_vecnormalize_path: Path | None = None
    if args.resume_from is not None:
        resume_model_path, resume_vecnormalize_path = find_resume_artifacts(run_dir)
        if (
            bool(config["normalize_observation"]) or bool(config["normalize_reward"])
        ) and resume_vecnormalize_path is None:
            raise FileNotFoundError(
                "Resume requires VecNormalize statistics because normalization is "
                "enabled. Check the run's models/ directory for "
                "vecnormalize_latest.pkl or checkpoint_model_vecnormalize_*_steps.pkl."
            )

    train_env = make_vector_env(
        env_id=config["env_id"],
        seed=seed,
        n_envs=int(config["n_envs"]),
        vec_env_type=config["vec_env"],
        monitor_dir=monitor_dir,
        normalize_observation=bool(config["normalize_observation"]),
        normalize_reward=bool(config["normalize_reward"]),
        norm_gamma=float(config["norm_gamma"]),
        clip_obs=float(config["clip_obs"]),
        vecnormalize_path=resume_vecnormalize_path,
        training=True,
    )
    eval_env = make_vector_env(
        env_id=config["env_id"],
        seed=seed + 100_000,
        n_envs=1,
        vec_env_type="dummy",
        monitor_dir=eval_dir,
        normalize_observation=bool(config["normalize_observation"]),
        normalize_reward=False,
        norm_gamma=float(config["norm_gamma"]),
        clip_obs=float(config["clip_obs"]),
        vecnormalize_path=resume_vecnormalize_path,
        training=False,
    )

    if resume_model_path is not None:
        model = PPO.load(
            str(resume_model_path),
            env=train_env,
            device=config["device"],
            seed=seed,
        )
        reset_num_timesteps = False
    else:
        model = build_model(config, train_env, logs_dir / "tensorboard")
        reset_num_timesteps = True

    model_steps = int(getattr(model, "num_timesteps", 0))
    metadata_path = run_dir / "metadata.json"
    previous_metadata = read_json(metadata_path, default={}) or {}
    metadata_steps = int(previous_metadata.get("last_num_timesteps", 0))
    if args.resume_from is not None and metadata_steps and model_steps != metadata_steps:
        raise ValueError(
            f"Resume artifact mismatch: model has {model_steps} steps but "
            f"metadata records {metadata_steps}. Use a consistent checkpoint pair."
        )
    current_steps = model_steps or metadata_steps
    target_steps = int(config["target_steps"])

    if target_steps > MAX_ENV_STEPS:
        raise ValueError(f"target_steps cannot exceed {MAX_ENV_STEPS}.")
    if current_steps > MAX_ENV_STEPS:
        raise ValueError(
            f"Recorded training steps {current_steps} already exceed assignment limit."
        )

    remaining_steps = target_steps - current_steps
    rollout_size = int(config["ppo"]["n_steps"]) * int(config["n_envs"])
    if remaining_steps > 0 and remaining_steps % rollout_size != 0:
        raise ValueError(
            f"remaining_steps={remaining_steps} is not divisible by PPO rollout_size="
            f"{rollout_size}. Use a target_steps value aligned to the rollout size "
            "so Stable-Baselines3 does not overshoot the assignment step limit."
        )
    write_config(run_dir / "config.json", config)

    metadata = {
        **previous_metadata,
        "run_dir": str(run_dir),
        "updated_at_utc": utc_now(),
        "runtime": runtime_info(Path.cwd()),
        "config": config,
        "command": " ".join(sys.argv),
        "assignment_limits": {
            "env_id": config["env_id"],
            "max_env_steps": MAX_ENV_STEPS,
            "raw_eval_reward_required": True,
        },
        "resume": args.resume_from is not None,
        "last_num_timesteps": current_steps,
        "target_steps": target_steps,
    }
    write_json(metadata_path, metadata)

    if remaining_steps <= 0:
        print(
            f"Run already has {current_steps} steps; target_steps={target_steps}. "
            "Nothing to train."
        )
    else:
        print(
            f"Training {remaining_steps} additional environment steps "
            f"({current_steps} -> {target_steps}) in {run_dir}"
        )

        n_envs = int(config["n_envs"])
        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=callback_freq(int(config["checkpoint_freq"]), n_envs),
                    save_path=str(models_dir),
                    name_prefix="checkpoint_model",
                    save_replay_buffer=False,
                    save_vecnormalize=True,
                ),
                SaveVecNormalizeCallback(
                    save_freq=callback_freq(int(config["checkpoint_freq"]), n_envs),
                    save_dir=models_dir,
                ),
                EvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=str(models_dir / "best"),
                    log_path=str(eval_dir),
                    eval_freq=callback_freq(int(config["eval_freq"]), n_envs),
                    n_eval_episodes=int(config["eval_episodes"]),
                    deterministic=True,
                    render=False,
                ),
                MetadataCallback(
                    metadata_path=metadata_path,
                    payload=metadata,
                    save_freq=callback_freq(int(config["checkpoint_freq"]), n_envs),
                ),
            ]
        )

        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=reset_num_timesteps,
            callback=callbacks,
            log_interval=int(config["log_interval"]),
            progress_bar=bool(args.progress_bar),
            tb_log_name="ppo_humanoid",
        )

    final_steps = int(model.num_timesteps)
    if final_steps > MAX_ENV_STEPS:
        raise RuntimeError(
            f"Training reached {final_steps} steps, exceeding assignment limit "
            f"{MAX_ENV_STEPS}."
        )
    model.save(str(models_dir / "latest_model.zip"))
    model.save(str(models_dir / f"final_model_{final_steps}_steps.zip"))

    if hasattr(train_env, "save"):
        train_env.save(str(models_dir / "vecnormalize_latest.pkl"))
        train_env.save(str(models_dir / f"vecnormalize_final_{final_steps}_steps.pkl"))

    metadata["updated_at_utc"] = utc_now()
    metadata["last_num_timesteps"] = final_steps
    metadata["model_artifacts"] = {
        "latest_model": str(models_dir / "latest_model.zip"),
        "latest_vecnormalize": str(models_dir / "vecnormalize_latest.pkl"),
    }
    write_json(metadata_path, metadata)

    train_env.close()
    eval_env.close()

    print(f"Saved model and normalization statistics under: {models_dir}")
    print(f"Total recorded environment steps: {final_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
