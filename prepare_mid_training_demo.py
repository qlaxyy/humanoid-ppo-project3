from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from humanoid_rl.artifacts import checkpoint_artifacts
from humanoid_rl.io import read_json, utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated run directory for recording a short middle-stage "
            "training continuation from an existing checkpoint."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("runs/local_sac_cpu_5m_seed3407"),
    )
    parser.add_argument("--source-step", type=int, default=3_500_000)
    parser.add_argument(
        "--demo-run",
        type=Path,
        default=Path("runs/local_sac_mid_demo_3500k_to_3600k"),
    )
    parser.add_argument("--target-steps", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run = args.source_run
    demo_run = args.demo_run
    source_step = int(args.source_step)
    target_steps = int(args.target_steps or source_step + 100_000)

    if not source_run.exists():
        raise FileNotFoundError(f"Source run not found: {source_run}")
    if demo_run.exists():
        raise FileExistsError(
            f"Demo run already exists: {demo_run}. Use a new --demo-run name."
        )
    if target_steps <= source_step:
        raise ValueError("--target-steps must be greater than --source-step.")

    model_path, vecnormalize_path = checkpoint_artifacts(
        source_run,
        source_step,
        require_vecnormalize=False,
    )
    source_config_path = source_run / "config.json"
    if not source_config_path.exists():
        raise FileNotFoundError(f"Source config not found: {source_config_path}")

    models_dir = demo_run / "models"
    for directory in [
        demo_run,
        models_dir,
        demo_run / "logs",
        demo_run / "monitor",
        demo_run / "evaluations",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    config = read_json(source_config_path)
    config["target_steps"] = target_steps
    config["output_dir"] = str(demo_run.parent)
    config["verbose"] = 1
    write_json(demo_run / "config.json", config)

    shutil.copy2(model_path, models_dir / model_path.name)
    if vecnormalize_path is not None:
        shutil.copy2(vecnormalize_path, models_dir / vecnormalize_path.name)

    source_metadata = read_json(source_run / "metadata.json", default={}) or {}
    write_json(
        demo_run / "metadata.json",
        {
            "run_dir": str(demo_run),
            "updated_at_utc": utc_now(),
            "last_num_timesteps": source_step,
            "target_steps": target_steps,
            "resume": True,
            "recording_demo": True,
            "source_run": str(source_run),
            "source_checkpoint_step": source_step,
            "source_checkpoint_model": str(model_path),
            "source_metadata_commit": source_metadata.get("runtime", {})
            .get("git", {})
            .get("commit"),
            "note": (
                "This isolated run is for recording a middle-stage training "
                "continuation. The final submission policy remains in the "
                "source run unless explicitly changed."
            ),
        },
    )

    print(f"Created demo run: {demo_run}")
    print(f"Copied checkpoint: {model_path}")
    print(f"Demo starts at: {source_step}")
    print(f"Demo target: {target_steps}")
    print()
    print("Record the middle-stage clip with:")
    print(
        "python train_sac.py "
        f"--resume-from {demo_run} "
        f"--resume-step {source_step} "
        f"--target-steps {target_steps} "
        "--device cpu --quiet --no-progress-bar --status-freq 5000 "
        "--checkpoint-freq 50000 --eval-freq 50000"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
