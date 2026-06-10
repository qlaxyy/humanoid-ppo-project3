# Final Candidate

Current best candidate:

```text
algorithm: SAC
run_id: local_sac_cpu_5m_seed3407
checkpoint_step: 5000000
config: configs/sac_humanoid_cpu_probe.json
training_limit: 5000000 environment steps
selected_actual_step: 5000000
normalization: none
```

Artifacts:

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
```

This SAC run does not use VecNormalize, so no normalization artifact is required
for inference.

Evaluation:

```text
seeds: 0 1 2 3 4 5 6 7 8 9
episodes_per_seed: 1
mean_reward: 6780.676
std_reward: 24.244
min_reward: 6753.711
max_reward: 6817.721
mean_length: 1000.0
seed_3407_reward: 6770.456
```

Checkpoint sweep:

```text
seeds: 0 1 2 3 4
best_step: 5000000
mean_reward: 6784.349
std_reward: 21.355
min_reward: 6757.746
max_reward: 6809.433
mean_length: 1000.0
```

Commands:

```bash
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 3407 --episodes 1 --device cpu
```

Previous candidate:

```text
algorithm: SAC
run_id: colab_sac_cpu_probe_200k_seed3407
checkpoint_step: 4000000
10-seed mean_reward: 6192.997
```

Earlier SAC candidate:

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

The local SAC 5.0M checkpoint replaces the SAC 4.0M, SAC 1.9M, SAC 900k, and PPO RL Zoo-style
candidates because its formal 10-seed raw reward evaluation is substantially
higher while staying below the assignment limit of 5,000,000 environment
interactions.
