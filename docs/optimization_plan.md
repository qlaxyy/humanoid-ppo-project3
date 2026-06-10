# Optimization Plan

Previous best candidate:

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

Observed checkpoint sweep result:

```text
best_step: 5000000
mean_reward: 2275.000
std_reward: 814.137
min_reward: 1679.109
max_reward: 3875.276
mean_length: 245.4
```

After the sweep, formally evaluate the best checkpoint:

```bash
python evaluate.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

If the formal 10-seed mean reward remains above `927.009`, update `docs/final_candidate.md` to the RL Zoo-style 5M checkpoint.

Formal 10-seed result:

```text
run_id: 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel
checkpoint_step: 5000000
mean_reward: 2179.016
std_reward: 666.613
min_reward: 1497.165
max_reward: 3875.276
mean_length: 237.8
seed_123_reward: 2596.866
```

Conclusion: this branch replaces the previous baseline checkpoint as the final candidate.

## Next Optional Branch: SAC

The next paper-reproduction branch is Soft Actor-Critic:

```text
configs/sac_humanoid_rlzoo.json
configs/sac_humanoid_obsnorm.json
```

Run only a 1,000,000-step comparison first. Continue a SAC branch to 5,000,000
steps only if its early result is competitive with the current final candidate.
See `docs/sac_and_video.md` for exact commands.

Observed CPU probe result:

```text
run_id: 20260605_153005_seed3407_sac_humanoid_cpu_probe
actual_steps: 200000
mean_reward: 896.367
std_reward: 302.616
min_reward: 560.438
max_reward: 1398.884
mean_length: 178.6
seed_123_reward: 952.200
```

Conclusion: SAC is promising at low step count, but the CPU-only training speed
is too slow for a long free-Colab run. Keep the PPO RL Zoo-style 5M checkpoint
as the final candidate unless a future GPU-backed SAC run exceeds `2179.016`.

Observed Kaggle fast-probe result:

```text
run_id: 20260606_035618_seed3407_sac_humanoid_fast_probe
platform: Kaggle T4 GPU
config: configs/sac_humanoid_fast_probe.json
actual_steps: 200000
mean_reward: 481.650
std_reward: 156.369
min_reward: 233.665
max_reward: 661.684
mean_length: 99.2
```

Conclusion: this faster SAC variant is useful for wall-clock testing but learns
too slowly under the current settings. Do not replace the PPO final candidate
or continue this branch unless a later configuration shows a much stronger
early score.
