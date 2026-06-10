# Final Candidate

Current best candidate:

```text
run_id: 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel
checkpoint_step: 5000000
config: configs/ppo_humanoid_rlzoo_parallel.json
training_limit: 5000000 environment steps
selected_actual_step: 5000000
```

Artifacts:

```text
runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel/models/checkpoint_model_5000000_steps.zip
runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel/models/checkpoint_model_vecnormalize_5000000_steps.pkl
```

Evaluation:

```text
seeds: 0 1 2 3 4 5 6 7 8 9
episodes_per_seed: 1
mean_reward: 2179.016
std_reward: 666.613
min_reward: 1497.165
max_reward: 3875.276
mean_length: 237.8
seed_123_reward: 2596.866
```

Checkpoint sweep:

```text
seeds: 0 1 2 3 4
best_step: 5000000
mean_reward: 2275.000
std_reward: 814.137
min_reward: 1679.109
max_reward: 3875.276
mean_length: 245.4
```

Commands:

```bash
python evaluate.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

Previous candidate:

```text
run_id: 20260605_031927_seed3407_ppo_humanoid_colab
checkpoint_step: 4500000
10-seed mean_reward: 927.009
```

The RL Zoo-style PPO candidate replaces the previous baseline checkpoint because its formal 10-seed raw reward evaluation is substantially higher.
