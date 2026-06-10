from __future__ import annotations

import re
from pathlib import Path


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else -1


def checkpoint_artifacts(
    run_dir: Path,
    step: int,
    require_vecnormalize: bool = True,
) -> tuple[Path, Path | None]:
    models_dir = run_dir / "models"
    model_path = models_dir / f"checkpoint_model_{step}_steps.zip"
    vecnormalize_candidates = [
        models_dir / f"checkpoint_model_vecnormalize_{step}_steps.pkl",
        models_dir / f"vecnormalize_{step}_steps.pkl",
    ]

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint model not found: {model_path}")

    vecnormalize_path = next(
        (path for path in vecnormalize_candidates if path.exists()),
        None,
    )
    if vecnormalize_path is None and require_vecnormalize:
        raise FileNotFoundError(
            "Checkpoint VecNormalize statistics not found. Tried: "
            + ", ".join(str(path) for path in vecnormalize_candidates)
        )

    return model_path, vecnormalize_path


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
    return model_path, vecnormalize_path
