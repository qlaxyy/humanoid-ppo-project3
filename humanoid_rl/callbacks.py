from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


def callback_freq(env_steps: int, n_envs: int) -> int:
    return max(1, int(env_steps) // max(1, int(n_envs)))


def find_vecnormalize(env) -> VecNormalize | None:
    current = env
    while current is not None:
        if isinstance(current, VecNormalize):
            return current
        current = getattr(current, "venv", None)
    return None


class SaveVecNormalizeCallback(BaseCallback):
    """Persist VecNormalize statistics during training and at the end."""

    def __init__(self, save_freq: int, save_dir: Path, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_freq = max(1, save_freq)
        self.save_dir = Path(save_dir)

    def _save(self) -> None:
        vecnormalize = find_vecnormalize(self.training_env)
        if vecnormalize is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        step_path = self.save_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
        latest_path = self.save_dir / "vecnormalize_latest.pkl"
        vecnormalize.save(str(step_path))
        vecnormalize.save(str(latest_path))

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save()
        return True

    def _on_training_end(self) -> None:
        self._save()


class MetadataCallback(BaseCallback):
    """Keep a small machine-readable run state for resume and reporting."""

    def __init__(self, metadata_path: Path, payload: dict[str, Any], save_freq: int):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.payload = payload
        self.save_freq = max(1, save_freq)

    def _write(self) -> None:
        data = dict(self.payload)
        data["last_num_timesteps"] = int(self.num_timesteps)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._write()
        return True

    def _on_training_end(self) -> None:
        self._write()

