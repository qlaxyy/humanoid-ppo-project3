# Optimization Plan

Current best candidate:

```text
run_id: 20260605_031927_seed3407_ppo_humanoid_colab
checkpoint_step: 4500000
10-seed mean_reward: 927.009
```

## Next Experiment

Try an RL Baselines3 Zoo style PPO configuration:

```text
configs/ppo_humanoid_rlzoo_parallel.json
```

Motivation:

- Lower learning rate than baseline
- ReLU activation instead of Tanh
- `ortho_init=false`
- `log_std_init=-2.0`
- `gamma=0.95`
- `gae_lambda=0.9`
- `clip_range=0.3`
- `n_epochs=5`

Run a 1,000,000-step comparison first:

```bash
python train.py --config configs/ppo_humanoid_rlzoo_parallel.json --target-steps 1000000
python evaluate.py --run-dir runs/<run_id> --seeds 0 1 2 3 4 --episodes-per-seed 1 --device cpu
```

Observed 1,000,000-step result:

```text
run_id: 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel
mean_reward: 886.012
std_reward: 149.381
mean_length: 144.4
seed_123_reward: 900.899
```

Decision rule:

- If 1M mean reward is clearly below the current baseline, stop this branch.
- If 1M mean reward is competitive, train to 5M and sweep checkpoints.
- Replace the final candidate only if the 10-seed checkpoint evaluation exceeds `927.009`.

The 1M result is competitive, so continue this branch:

```bash
python train.py --resume-from runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --target-steps 5000000 --device cpu
python evaluate_checkpoints.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --every 500000 --seeds 0 1 2 3 4 --device cpu
```

After the sweep, formally evaluate the best checkpoint:

```bash
python evaluate.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step <best_step> --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step <best_step> --seed 123 --episodes 1 --device cpu
```
