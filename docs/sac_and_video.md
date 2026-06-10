# SAC and Video Workflow

## Why Try SAC

SAC is an off-policy maximum-entropy algorithm for continuous control. It is a
reasonable paper-reproduction branch after the PPO and RL Zoo-style PPO
experiments.

The current final candidate remains:

```text
run_id: 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel
checkpoint_step: 5000000
10-seed mean_reward: 2179.016
```

Run SAC only as a new independent branch. Do not continue training any submitted
candidate beyond 5,000,000 environment steps.

## SAC Short Runs

RL Zoo-style SAC branch:

```bash
python train_sac.py --config configs/sac_humanoid_rlzoo.json --target-steps 1000000 --device auto
python evaluate.py --run-dir runs/<sac_run_name> --seeds 0 1 2 3 4 --episodes-per-seed 1 --device cpu
```

Observation-normalized SAC branch:

```bash
python train_sac.py --config configs/sac_humanoid_obsnorm.json --target-steps 1000000 --device auto
python evaluate.py --run-dir runs/<sac_obsnorm_run_name> --seeds 0 1 2 3 4 --episodes-per-seed 1 --device cpu
```

Continue a SAC branch only if its 1M result is competitive with the current
candidate:

```bash
python train_sac.py --resume-from runs/<sac_run_name> --target-steps 5000000 --device auto
python evaluate_checkpoints.py --run-dir runs/<sac_run_name> --every 500000 --seeds 0 1 2 3 4 --device cpu
```

Then formally evaluate the best checkpoint:

```bash
python evaluate.py --run-dir runs/<sac_run_name> --checkpoint-step <best_step> --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/<sac_run_name> --checkpoint-step <best_step> --seed 123 --episodes 1 --device cpu
```

Note: SAC replay buffers are not saved by default because a Humanoid replay
buffer can become very large. Resuming SAC continues from the policy checkpoint
but starts with an empty replay buffer, so uninterrupted runs are preferred.

## Recording Videos

The Gymnasium website is documentation only. Videos are generated locally by
loading the trained model, rendering `Humanoid-v5`, and saving an mp4 file.

For the current final candidate:

```bash
python record_video.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

On Colab, MuJoCo rendering should use a headless OpenGL backend. First test
whether the runtime can render one Humanoid frame:

```bash
python render_probe.py --backend egl --output videos/render_probe_egl.png
```

If that works, record a short model video:

```bash
python record_video.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu --max-steps 80
```

The script writes mp4 files to a local `/content` temporary directory first,
then copies the finished file to Drive. This avoids hangs caused by streaming
ffmpeg output directly into Google Drive.

If that works, record the full episode:

```bash
python record_video.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

If EGL still hangs, try OSMesa after installing its runtime package in Colab:

```bash
apt-get update
apt-get install -y libosmesa6
MUJOCO_GL=osmesa python render_probe.py --backend osmesa --output videos/render_probe_osmesa.png
MUJOCO_GL=osmesa python record_video.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu --backend osmesa --max-steps 80
```

If mp4 writing still fails, save PNG frames only:

```bash
python record_video.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu --max-steps 80 --frames-only
```

If rendering hangs only after loading the SB3 model, split policy execution and
rendering into two processes:

```bash
python export_trajectory.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu --max-steps 80
python render_trajectory.py --trajectory runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel/trajectories/<trajectory_file>.npz --backend egl
```

The second script does not import Stable-Baselines3 or load the neural-network
model. It only replays the saved action sequence, which avoids model/rendering
backend conflicts on Colab.

The 80-step probe is expected to be short: at 30 fps it lasts about 2.7 seconds.
For a full policy episode, export without `--max-steps`:

```bash
python export_trajectory.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
python render_latest_trajectory.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --backend egl --fps 10
```

For a longer visual demonstration, repeat the latest exported trajectory:

```bash
python render_latest_trajectory.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --backend egl --fps 10 --repeat 3
```

For the assignment's requested five-minute process videos, screen-recording the
Colab notebook while running the training/evaluation commands is more suitable
than trying to stretch a short environment episode to five minutes.

General form:

```bash
python record_video.py --run-dir runs/<run_name> --checkpoint-step <best_step> --seed 123 --episodes 1 --device cpu
```

The script saves videos under:

```text
runs/<run_name>/videos/
```
