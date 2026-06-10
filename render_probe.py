from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def requested_backend(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg == "--backend" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1]
    return None


os.environ.setdefault("MUJOCO_GL", requested_backend(sys.argv) or "egl")

import gymnasium as gym
import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Gymnasium MuJoCo rendering.")
    parser.add_argument("--env-id", type=str, default="Humanoid-v5")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", choices=["egl", "osmesa", "glfw"], default=None)
    parser.add_argument("--output", type=Path, default=Path("videos/render_probe.png"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    print(f"Creating {args.env_id} with render_mode=rgb_array", flush=True)
    env = gym.make(args.env_id, render_mode="rgb_array")
    try:
        print(f"Resetting with seed={args.seed}", flush=True)
        env.reset(seed=args.seed)
        print("Rendering one frame", flush=True)
        frame = env.render()
        print(f"Frame shape: {getattr(frame, 'shape', None)}", flush=True)
        imageio.imwrite(args.output, frame)
        print(f"Saved probe image: {args.output}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
