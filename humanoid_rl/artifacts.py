from __future__ import annotations

from pathlib import Path


def checkpoint_artifacts(run_dir: Path, step: int) -> tuple[Path, Path]:
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
    if vecnormalize_path is None:
        raise FileNotFoundError(
            "Checkpoint VecNormalize statistics not found. Tried: "
            + ", ".join(str(path) for path in vecnormalize_candidates)
        )

    return model_path, vecnormalize_path

