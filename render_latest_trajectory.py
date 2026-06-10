from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the latest exported trajectory under a run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["egl", "osmesa", "glfw"], default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--frames-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory_dir = args.run_dir / "trajectories"
    trajectories = sorted(trajectory_dir.glob("*.npz"))
    if not trajectories:
        raise FileNotFoundError(f"No .npz trajectories found under {trajectory_dir}")

    trajectory = trajectories[-1]
    print(f"Rendering latest trajectory: {trajectory}", flush=True)
    command = [
        sys.executable,
        "render_trajectory.py",
        "--trajectory",
        str(trajectory),
        "--fps",
        str(args.fps),
        "--repeat",
        str(args.repeat),
    ]
    if args.backend is not None:
        command.extend(["--backend", args.backend])
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.frames_only:
        command.append("--frames-only")

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
