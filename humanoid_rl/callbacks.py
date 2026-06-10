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
        metric_table: bool = False,
    ):
        super().__init__()
        self.target_steps = int(target_steps)
        self.start_steps = int(start_steps)
        self.report_freq = max(1, int(report_freq))
        self.status_path = Path(status_path)
        self.metric_table = bool(metric_table)
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

    @staticmethod
    def _mean(values: deque[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _format_metric(key: str, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

        integer_like_keys = (
            "episodes",
            "time_elapsed",
            "total_timesteps",
            "n_updates",
        )
        if any(key.endswith(name) for name in integer_like_keys):
            return str(int(round(numeric)))
        if key.endswith("fps"):
            return f"{numeric:.1f}"
        if key.endswith("ep_len_mean"):
            return f"{numeric:.1f}"
        if key.endswith("ep_rew_mean"):
            return f"{numeric:.3f}"
        if key.endswith("learning_rate"):
            return f"{numeric:.6f}"
        return f"{numeric:.3f}"

    def _latest_logger_values(self) -> dict[str, Any]:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return {}
        values = getattr(logger, "name_to_value", {})
        return dict(values) if isinstance(values, dict) else {}

    def _print_metric_table(
        self,
        *,
        elapsed: float,
        fps: float,
        recent_reward: float | None,
        recent_length: float | None,
    ) -> None:
        logger_values = self._latest_logger_values()
        train_values = {
            "actor_loss": logger_values.get("train/actor_loss"),
            "critic_loss": logger_values.get("train/critic_loss"),
            "ent_coef": logger_values.get("train/ent_coef"),
            "ent_coef_loss": logger_values.get("train/ent_coef_loss"),
            "learning_rate": logger_values.get("train/learning_rate"),
            "n_updates": getattr(self.model, "_n_updates", None),
        }
        sections: list[tuple[str, list[tuple[str, Any]]]] = [
            (
                "rollout/",
                [
                    ("ep_len_mean", recent_length),
                    ("ep_rew_mean", recent_reward),
                ],
            ),
            (
                "time/",
                [
                    ("episodes", getattr(self.model, "_episode_num", None)),
                    ("fps", fps),
                    ("time_elapsed", int(elapsed)),
                    ("total_timesteps", int(self.num_timesteps)),
                ],
            ),
            (
                "train/",
                [
                    (key, value)
                    for key, value in train_values.items()
                    if value is not None
                ],
            ),
        ]

        rows: list[tuple[str, str]] = []
        for section, metrics in sections:
            rows.append((section, ""))
            for key, value in metrics:
                rows.append(
                    (
                        f"    {key}",
                        self._format_metric(f"{section}{key}", value),
                    )
                )

        key_width = max(len(key) for key, _ in rows)
        value_width = max(12, max(len(value) for _, value in rows))
        border = "-" * (key_width + value_width + 7)
        print(border, flush=True)
        for key, value in rows:
            print(
                f"| {key:<{key_width}} | {value:<{value_width}} |",
                flush=True,
            )
        print(border, flush=True)

    def _report(self) -> None:
        elapsed = max(1e-9, time.monotonic() - self.started_at)
        trained_steps = max(0, int(self.num_timesteps) - self.start_steps)
        fps = trained_steps / elapsed if trained_steps else 0.0
        remaining_steps = max(0, self.target_steps - int(self.num_timesteps))
        eta_seconds = remaining_steps / fps if fps > 0 else 0.0
        percent = 100.0 * int(self.num_timesteps) / max(1, self.target_steps)
        recent_reward = self._mean(self.episode_rewards)
        recent_length = self._mean(self.episode_lengths)
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
        if self.metric_table:
            self._print_metric_table(
                elapsed=elapsed,
                fps=fps,
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
