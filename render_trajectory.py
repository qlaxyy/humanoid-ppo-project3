from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def requested_backend(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg == "--backend" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1]
    return None


def default_backend() -> str:
    if sys.platform.startswith("win"):
        return "glfw"
    return "egl"


def resolved_backend(argv: list[str]) -> str:
    backend = requested_backend(argv) or default_backend()
    if sys.platform.startswith("win") and backend == "egl":
        print("Windows detected; falling back from egl to glfw.", flush=True)
        return "glfw"
    return backend


os.environ.setdefault("MUJOCO_GL", resolved_backend(sys.argv))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an exported action trajectory without loading SB3."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["egl", "osmesa", "glfw"], default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--local-video-workdir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = np.load(args.trajectory, allow_pickle=False)
    env_id = str(data["env_id"])
    seed = int(data["seed"])
    actions = np.asarray(data["actions"], dtype=np.float32)

    video_dir = args.video_dir or args.trajectory.parent / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    local_video_dir = Path(
        args.local_video_workdir
        or tempfile.mkdtemp(prefix="humanoid_video_", dir="/content" if Path("/content").exists() else None)
    )
    local_video_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.trajectory.stem

    print(f"Using MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    print(f"Loaded {len(actions)} actions from {args.trajectory}", flush=True)
    print(f"Creating {env_id} with render_mode=rgb_array", flush=True)
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)

        local_video_path = local_video_dir / f"{prefix}.mp4"
        final_video_path = video_dir / local_video_path.name

        writer = None
        if args.frames_only:
            print("Rendering PNG frames only; mp4 writer is disabled.", flush=True)
        else:
            writer = imageio.get_writer(
                str(local_video_path),
                fps=int(args.fps),
                macro_block_size=1,
            )

        total_reward = 0.0
        length = 0
        terminated = False
        truncated = False
        try:
            print("Rendering first frame", flush=True)
            frame = env.render()
            if writer is not None:
                writer.append_data(frame)
            if args.frames_only:
                imageio.imwrite(video_dir / f"{prefix}-frame-0000.png", frame)

            repeat_count = max(1, int(args.repeat))
            for repeat_index in range(repeat_count):
                if repeat_index > 0:
                    env.reset(seed=seed)
                    frame = env.render()
                    if writer is not None:
                        writer.append_data(frame)
                    if args.frames_only:
                        imageio.imwrite(video_dir / f"{prefix}-frame-{length:04d}.png", frame)

                for action in actions:
                    if args.max_steps is not None and length >= int(args.max_steps):
                        truncated = True
                        break
                    _, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    length += 1
                    frame = env.render()
                    if writer is not None:
                        writer.append_data(frame)
                    if args.frames_only:
                        imageio.imwrite(video_dir / f"{prefix}-frame-{length:04d}.png", frame)
                    if length % 50 == 0:
                        print(f"  rendered {length} steps", flush=True)
                    if terminated or truncated:
                        break

                if terminated or truncated or (
                    args.max_steps is not None and length >= int(args.max_steps)
                ):
                    break
        finally:
            if writer is not None:
                writer.close()

        if not args.frames_only:
            print(f"Copying video to {final_video_path}", flush=True)
            shutil.copy2(local_video_path, final_video_path)
            print(f"Saved video: {final_video_path}", flush=True)
        print(
            f"Replay reward={total_reward:.3f} length={length} "
            f"terminated={terminated} truncated={truncated}",
            flush=True,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
