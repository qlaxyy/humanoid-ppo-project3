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

Observed Kaggle rerun of the Colab SAC cpu-probe config:

```text
run_id: kaggle_sac_cpu_probe_200k_seed3407
platform: Kaggle T4 GPU
config: configs/sac_humanoid_cpu_probe.json
actual_steps: 200000
mean_reward: 528.677
std_reward: 41.109
min_reward: 468.050
max_reward: 590.816
mean_length: 103.2
seed_123_reward: 533.753
```

Conclusion: the Kaggle rerun did not reproduce the stronger Colab SAC probe
score. Because SAC is more sensitive to device/platform nondeterminism and
the result is still far below the PPO final candidate, pause the SAC branch
unless there is enough time for multiple seeds or a longer controlled run.

Observed Colab rerun of the SAC cpu-probe config:

```text
run_id: colab_sac_cpu_probe_200k_seed3407
platform: Colab CPU
config: configs/sac_humanoid_cpu_probe.json
actual_steps: 200000
mean_reward: 579.108
std_reward: 55.303
min_reward: 477.429
max_reward: 638.149
mean_length: 113.4
seed_123_reward: 620.384
```

Later continuation to 1M:

```text
run_id: colab_sac_cpu_probe_200k_seed3407
platform: Colab CPU
config: configs/sac_humanoid_cpu_probe.json
actual_steps: 1000000
mean_reward: 1520.323
std_reward: 909.567
min_reward: 465.276
max_reward: 2628.132
mean_length: 282.0
seed_123_reward: 2816.869
```

Conclusion: SAC improved substantially after 1M steps, but the 5-seed mean is
still below the PPO RL Zoo-style final candidate (`2179.016`) and the variance
is high. Sweep SAC checkpoints before any longer run; continue to 5M only if a
checkpoint evaluation is clearly competitive.

Checkpoint sweep after the 1M SAC run:

```text
best_step: 900000
mean_reward: 5421.415
std_reward: 13.155
min_reward: 5403.780
max_reward: 5440.614
mean_length: 1000.0
```

Formal 10-seed evaluation of the 900k checkpoint:

```text
run_id: colab_sac_cpu_probe_200k_seed3407
checkpoint_step: 900000
mean_reward: 5379.517
std_reward: 129.155
min_reward: 4993.352
max_reward: 5440.614
mean_length: 993.6
seed_123_reward: 4472.476
```

Conclusion: the SAC 900k checkpoint replaces the PPO RL Zoo-style checkpoint
as the final candidate. It is both higher scoring and trained with only 900k
environment interactions, well below the 5M assignment limit.

Continuation to 2M found a stronger SAC checkpoint:

```text
run_id: colab_sac_cpu_probe_200k_seed3407
checkpoint_step: 1900000
platform: Colab CPU
config: configs/sac_humanoid_cpu_probe.json
mean_reward: 5983.969
std_reward: 25.421
min_reward: 5943.729
max_reward: 6027.610
mean_length: 1000.0
seed_123_reward: 5997.139
```

Conclusion: the SAC 1.9M checkpoint replaces the SAC 900k checkpoint as the
final candidate. It is stable across the 10 evaluation seeds, reaches the
1000-step episode limit on every evaluated episode, and remains far below the
5M environment interaction limit.

Continuation to 5M produced a stronger checkpoint sweep candidate:

```text
run_id: colab_sac_cpu_probe_200k_seed3407
target_steps: 5000000
best_checkpoint_step: 4000000
platform: Colab CPU
config: configs/sac_humanoid_cpu_probe.json
mean_reward: 6217.259
std_reward: 32.654
min_reward: 6154.199
max_reward: 6247.676
mean_length: 1000.0
```

Conclusion: the SAC 4.0M checkpoint is the best checkpoint from the 5M sweep
reported so far and should receive a formal 10-seed evaluation. Do not replace
the SAC 1.9M final candidate until that formal evaluation confirms the gain.
