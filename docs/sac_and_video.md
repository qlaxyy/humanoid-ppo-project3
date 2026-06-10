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

On Colab, MuJoCo rendering should use a headless OpenGL backend. The script sets
`MUJOCO_GL=egl` by default before importing Gymnasium. If rendering still fails,
run the same command with the environment variable explicitly set:

```bash
MUJOCO_GL=egl python record_video.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

General form:

```bash
python record_video.py --run-dir runs/<run_name> --checkpoint-step <best_step> --seed 123 --episodes 1 --device cpu
```

The script saves videos under:

```text
runs/<run_name>/videos/
```
