from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


TRACKED_PACKAGES = [
    "gymnasium",
    "mujoco",
    "stable-baselines3",
    "torch",
    "numpy",
    "cloudpickle",
    "pandas",
    "matplotlib",
    "tensorboard",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_for_path() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_info(cwd: Path) -> dict[str, str | bool]:
    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip()

    commit = run_git(["rev-parse", "--short", "HEAD"]) or "uncommitted"
    branch = run_git(["branch", "--show-current"]) or "unknown"
    dirty = bool(run_git(["status", "--short"]))
    return {"commit": commit, "branch": branch, "dirty": dirty}


def runtime_info(cwd: Path) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": git_info(cwd),
        "torch": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        },
    }

