from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot checkpoint sweep mean reward over training steps."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/checkpoint_sweep_data.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/checkpoint_sweep_curve.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": int(row["step"]),
                    "mean_reward": float(row["mean_reward"]),
                    "std_reward": float(row["std_reward"]),
                    "mean_length": float(row["mean_length"]),
                }
            )

    rows.sort(key=lambda row: row["step"])
    steps_m = [row["step"] / 1_000_000 for row in rows]
    means = [row["mean_reward"] for row in rows]
    stds = [row["std_reward"] for row in rows]
    lower = [mean - std for mean, std in zip(means, stds)]
    upper = [mean + std for mean, std in zip(means, stds)]

    best = max(rows, key=lambda row: row["mean_reward"])
    best_step_m = best["step"] / 1_000_000

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    ax.plot(
        steps_m,
        means,
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="Mean raw reward",
    )
    ax.fill_between(
        steps_m,
        lower,
        upper,
        color="#1f77b4",
        alpha=0.12,
        linewidth=0,
        label="Mean +/- std",
    )
    ax.scatter(
        [best_step_m],
        [best["mean_reward"]],
        color="#d62728",
        s=55,
        zorder=5,
        label=f"Best: {best['step']:,} steps",
    )
    ax.annotate(
        f"{best['mean_reward']:.1f}",
        xy=(best_step_m, best["mean_reward"]),
        xytext=(-48, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0},
        fontsize=9,
    )

    ax.set_title("Checkpoint Sweep on Humanoid-v5", fontsize=14, pad=12)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Raw episode reward")
    ax.set_xlim(0, 5.15)
    ax.set_ylim(0, max(upper) * 1.08)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
