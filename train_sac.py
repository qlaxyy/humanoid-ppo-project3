from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from humanoid_rl import MAX_ENV_STEPS
from humanoid_rl.artifacts import find_resume_artifacts
from humanoid_rl.callbacks import (
    MetadataCallback,
    ProgressReporterCallback,
    SaveVecNormalizeCallback,
    callback_freq,
)
from humanoid_rl.config import load_config, validate_config, write_config
from humanoid_rl.envs import make_vector_env
from humanoid_rl.io import read_json, runtime_info, timestamp_for_path, utc_now, write_json
from humanoid_rl.reproducibility import configure_torch_threads, seed_everything


DEFAULT_CONFIG_PATH = Path("configs/sac_humanoid_rlzoo.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on Gymnasium Humanoid-v5 with reproducible logging."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--status-freq",
        type=int,
        default=None,
        help="Print one compact progress line every N environment steps.",
    )

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
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
        "device": args.device,
        "torch_threads": args.torch_threads,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "log_interval": args.log_interval,
    }
    for key, value in direct_overrides.items():
        if value is not None:
            config[key] = value

    sac_overrides = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "gamma": args.gamma,
        "tau": args.tau,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
    }
    for key, value in sac_overrides.items():
        if value is not None:
            config["sac"][key] = value

    config["algorithm"] = "sac"
    config["n_envs"] = 1
    config["vec_env"] = "dummy"
    if args.quiet:
        config["verbose"] = 0
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
    config_label = (args.config.stem if args.config is not None else "sac_humanoid")
    run_name = args.run_name or f"{timestamp_for_path()}_seed{config['seed']}_{config_label}"
    run_dir = output_dir / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def coerce_ent_coef(value: Any) -> str | float:
    if isinstance(value, str):
        return value
    return float(value)


def build_model(config: dict[str, Any], train_env, tensorboard_dir: Path) -> SAC:
    sac = config["sac"]
    policy_kwargs = {
        "net_arch": {
            "pi": list(sac["policy_net_arch"]),
            "qf": list(sac["qf_net_arch"]),
        }
    }

    return SAC(
        policy=sac["policy"],
        env=train_env,
        learning_rate=make_schedule(sac["learning_rate"]),
        buffer_size=int(sac["buffer_size"]),
        learning_starts=int(sac["learning_starts"]),
        batch_size=int(sac["batch_size"]),
        tau=float(sac["tau"]),
        gamma=float(sac["gamma"]),
        train_freq=int(sac["train_freq"]),
        gradient_steps=int(sac["gradient_steps"]),
        ent_coef=coerce_ent_coef(sac["ent_coef"]),
        target_update_interval=int(sac["target_update_interval"]),
        target_entropy=sac["target_entropy"],
        use_sde=bool(sac["use_sde"]),
        policy_kwargs=policy_kwargs,
        seed=int(config["seed"]),
        device=config["device"],
        tensorboard_log=str(tensorboard_dir),
        verbose=int(config.get("verbose", 1)),
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
                "Resume requires VecNormalize statistics because normalization is enabled."
            )

    train_env = make_vector_env(
        env_id=config["env_id"],
        seed=seed,
        n_envs=1,
        vec_env_type="dummy",
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
        model = SAC.load(
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
            f"metadata records {metadata_steps}."
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

        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=int(config["checkpoint_freq"]),
                    save_path=str(models_dir),
                    name_prefix="checkpoint_model",
                    save_replay_buffer=False,
                    save_vecnormalize=True,
                ),
                SaveVecNormalizeCallback(
                    save_freq=int(config["checkpoint_freq"]),
                    save_dir=models_dir,
                ),
                EvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=str(models_dir / "best"),
                    log_path=str(eval_dir),
                    eval_freq=int(config["eval_freq"]),
                    n_eval_episodes=int(config["eval_episodes"]),
                    deterministic=True,
                    render=False,
                ),
                MetadataCallback(
                    metadata_path=metadata_path,
                    payload=metadata,
                    save_freq=int(config["checkpoint_freq"]),
                ),
            ]
        )
        if args.status_freq is not None and args.status_freq > 0:
            callbacks.callbacks.append(
                ProgressReporterCallback(
                    target_steps=target_steps,
                    start_steps=current_steps,
                    report_freq=int(args.status_freq),
                    status_path=run_dir / "progress.json",
                )
            )

        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=reset_num_timesteps,
            callback=callbacks,
            log_interval=int(config["log_interval"]),
            progress_bar=bool(args.progress_bar),
            tb_log_name="sac_humanoid",
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
