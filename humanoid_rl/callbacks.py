from __future__ import annotations

import json
import time
from collections import deque
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


class ProgressReporterCallback(BaseCallback):
    """Print compact progress lines and persist the latest status as JSON."""

    def __init__(
        self,
        target_steps: int,
        start_steps: int,
        report_freq: int,
        status_path: Path,
        reward_window: int = 20,
    ):
        super().__init__()
        self.target_steps = int(target_steps)
        self.start_steps = int(start_steps)
        self.report_freq = max(1, int(report_freq))
        self.status_path = Path(status_path)
        self.started_at = time.monotonic()
        self.last_report_steps = self.start_steps
        self.episode_rewards: deque[float] = deque(maxlen=max(1, int(reward_window)))
        self.episode_lengths: deque[float] = deque(maxlen=max(1, int(reward_window)))

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:d}:{secs:02d}"

    def _write_status(
        self,
        fps: float,
        eta_seconds: float,
        recent_reward: float | None,
        recent_length: float | None,
    ) -> None:
        payload = {
            "num_timesteps": int(self.num_timesteps),
            "target_steps": self.target_steps,
            "progress_percent": (
                100.0 * int(self.num_timesteps) / max(1, self.target_steps)
            ),
            "fps": fps,
            "eta_seconds": eta_seconds,
            "recent_ep_rew_mean": recent_reward,
            "recent_ep_len_mean": recent_length,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with self.status_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def _collect_episode_info(self) -> None:
        for info in self.locals.get("infos", []):
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode is None:
                continue
            if "r" in episode:
                self.episode_rewards.append(float(episode["r"]))
            if "l" in episode:
                self.episode_lengths.append(float(episode["l"]))

    def _report(self) -> None:
        elapsed = max(1e-9, time.monotonic() - self.started_at)
        trained_steps = max(0, int(self.num_timesteps) - self.start_steps)
        fps = trained_steps / elapsed if trained_steps else 0.0
        remaining_steps = max(0, self.target_steps - int(self.num_timesteps))
        eta_seconds = remaining_steps / fps if fps > 0 else 0.0
        percent = 100.0 * int(self.num_timesteps) / max(1, self.target_steps)
        recent_reward = (
            sum(self.episode_rewards) / len(self.episode_rewards)
            if self.episode_rewards
            else None
        )
        recent_length = (
            sum(self.episode_lengths) / len(self.episode_lengths)
            if self.episode_lengths
            else None
        )
        reward_text = (
            f" recent_ep_rew_mean={recent_reward:.3f}"
            f" recent_ep_len_mean={recent_length:.1f}"
            if recent_reward is not None and recent_length is not None
            else ""
        )
        print(
            "progress "
            f"{int(self.num_timesteps)}/{self.target_steps} "
            f"({percent:.1f}%) "
            f"fps={fps:.1f} "
            f"eta={self._format_seconds(eta_seconds)}"
            f"{reward_text}",
            flush=True,
        )
        self._write_status(
            fps=fps,
            eta_seconds=eta_seconds,
            recent_reward=recent_reward,
            recent_length=recent_length,
        )

    def _on_step(self) -> bool:
        self._collect_episode_info()
        current_steps = int(self.num_timesteps)
        if current_steps - self.last_report_steps >= self.report_freq:
            self.last_report_steps = current_steps
            self._report()
        return True

    def _on_training_end(self) -> None:
        self._report()
