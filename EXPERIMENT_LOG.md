# Experiment Log

所有正式或半正式实验都在这里登记。脚本会自动保存机器可读日志，但这张表方便最后写报告。

| Run ID | Date | Git Commit | Config | Seed | Target Steps | Actual Steps | n_envs | n_steps | Batch | Device | Mean Raw Reward | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| template | YYYY-MM-DD | `<commit>` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 |  | 4 | 1250 | 250 | Colab GPU / CPU |  |  |
| 20260604_091720_seed3407_ppo_humanoid | 2026-06-04 | `5d2748f` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab GPU | 833.358 | Baseline PPO, seeds 0-4, mean length 156.2 |
| 20260604_115101_seed3407_ppo_humanoid_colab | 2026-06-04 | `09e22f5` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab GPU | 850.491 | Baseline PPO rerun, seeds 0-4, std 204.609, mean length 161.0, seed 123 reward 1239.427 |
| 20260604_140725_seed3407_ppo_humanoid_stable | 2026-06-04 | `3cd7974` | `configs/ppo_humanoid_stable.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab T4 GPU | 753.044 | Stable PPO, seeds 0-4, std 95.806, mean length 135.0, seed 123 reward 818.042; worse than baseline |
| 20260605_031927_seed3407_ppo_humanoid_colab | 2026-06-05 | `73afbd7` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab T4 GPU | 833.358 | Baseline PPO in Drive-cloned repo, seeds 0-4, std 110.924, mean length 156.2, seed 123 reward 449.272 |

## 记录规范

- 每次改算法参数前先提交 Git，再训练。
- Run ID 使用 `runs/` 下的目录名。
- `Actual Steps` 以 `runs/<run_id>/metadata.json` 的 `last_num_timesteps` 为准。
- `Mean Raw Reward` 使用 `evaluate.py` 输出的 `mean_reward`，不要使用训练中的 normalized reward。
- 备注里记录是否断点续训、Colab 是否中断、是否出现依赖或环境异常。
