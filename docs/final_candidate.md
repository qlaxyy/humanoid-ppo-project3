# Final Candidate

Current best candidate:

```text
run_id: 20260605_031927_seed3407_ppo_humanoid_colab
checkpoint_step: 4500000
config: configs/ppo_humanoid_colab.json
training_limit: 5000000 environment steps
selected_actual_step: 4500000
```

Artifacts:

```text
runs/20260605_031927_seed3407_ppo_humanoid_colab/models/checkpoint_model_4500000_steps.zip
runs/20260605_031927_seed3407_ppo_humanoid_colab/models/checkpoint_model_vecnormalize_4500000_steps.pkl
```

Evaluation:

```text
seeds: 0 1 2 3 4 5 6 7 8 9
episodes_per_seed: 1
mean_reward: 927.009
std_reward: 173.737
min_reward: 629.626
max_reward: 1244.171
mean_length: 186.5
seed_123_reward: 1331.420
```

Commands:

```bash
python evaluate.py --run-dir runs/20260605_031927_seed3407_ppo_humanoid_colab --checkpoint-step 4500000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/20260605_031927_seed3407_ppo_humanoid_colab --checkpoint-step 4500000 --seed 123 --episodes 1 --device cpu
```

Important: do not submit `latest_model.zip` as the final candidate for this run, because the 5,000,000-step latest model evaluated worse than the 4,500,000-step checkpoint.

Further optimization candidates should be compared against this checkpoint before replacing it.
