# Final Candidate

Current best candidate:

```text
algorithm: SAC
run_id: colab_sac_cpu_probe_200k_seed3407
checkpoint_step: 4000000
config: configs/sac_humanoid_cpu_probe.json
training_limit: 5000000 environment steps
selected_actual_step: 4000000
normalization: none
```

Artifacts:

```text
runs/colab_sac_cpu_probe_200k_seed3407/models/checkpoint_model_4000000_steps.zip
```

This SAC run does not use VecNormalize, so no normalization artifact is required
for inference.

Evaluation:

```text
seeds: 0 1 2 3 4 5 6 7 8 9
episodes_per_seed: 1
mean_reward: 6192.997
std_reward: 44.335
min_reward: 6115.738
max_reward: 6247.676
mean_length: 1000.0
seed_123_reward: 6167.959
```

Checkpoint sweep:

```text
seeds: 0 1 2 3 4
best_step: 4000000
mean_reward: 6217.259
std_reward: 32.654
min_reward: 6154.199
max_reward: 6247.676
mean_length: 1000.0
```

Commands:

```bash
python evaluate.py --run-dir runs/colab_sac_cpu_probe_200k_seed3407 --checkpoint-step 4000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/colab_sac_cpu_probe_200k_seed3407 --checkpoint-step 4000000 --seed 123 --episodes 1 --device cpu
```

Previous candidate:

```text
algorithm: SAC
run_id: colab_sac_cpu_probe_200k_seed3407
checkpoint_step: 1900000
10-seed mean_reward: 5983.969
```

Earlier SAC candidate:

```text
algorithm: SAC
run_id: colab_sac_cpu_probe_200k_seed3407
checkpoint_step: 900000
10-seed mean_reward: 5379.517
```

Earlier PPO candidate:

```text
algorithm: PPO
run_id: 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel
checkpoint_step: 5000000
10-seed mean_reward: 2179.016
```

The SAC 4.0M checkpoint replaces the SAC 1.9M, SAC 900k, and PPO RL Zoo-style
candidates because its formal 10-seed raw reward evaluation is substantially
higher while staying below the assignment limit of 5,000,000 environment
interactions.
