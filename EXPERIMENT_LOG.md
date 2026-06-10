# Experiment Log

所有正式或半正式实验都在这里登记。脚本会自动保存机器可读日志，但这张表方便最后写报告。

| Run ID | Date | Git Commit | Config | Seed | Target Steps | Actual Steps | n_envs | n_steps | Batch | Device | Mean Raw Reward | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| template | YYYY-MM-DD | `<commit>` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 |  | 4 | 1250 | 250 | Colab GPU / CPU |  |  |
| 20260604_091720_seed3407_ppo_humanoid | 2026-06-04 | `5d2748f` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab GPU | 833.358 | Baseline PPO, seeds 0-4, mean length 156.2 |
| 20260604_115101_seed3407_ppo_humanoid_colab | 2026-06-04 | `09e22f5` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab GPU | 850.491 | Baseline PPO rerun, seeds 0-4, std 204.609, mean length 161.0, seed 123 reward 1239.427 |
| 20260604_140725_seed3407_ppo_humanoid_stable | 2026-06-04 | `3cd7974` | `configs/ppo_humanoid_stable.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab T4 GPU | 753.044 | Stable PPO, seeds 0-4, std 95.806, mean length 135.0, seed 123 reward 818.042; worse than baseline |
| 20260605_031927_seed3407_ppo_humanoid_colab | 2026-06-05 | `73afbd7` | `configs/ppo_humanoid_colab.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab T4 GPU | 833.358 | Baseline PPO in Drive-cloned repo, seeds 0-4, std 110.924, mean length 156.2, seed 123 reward 449.272 |
| 20260605_031927_seed3407_ppo_humanoid_colab | 2026-06-05 | `73afbd7` | `configs/ppo_humanoid_colab.json` | 3407 | 5000000 | 5000000 | 4 | 1250 | 250 | Colab T4 GPU | 714.891 | Same baseline continued to 5M, seeds 0-4, std 73.373, mean length 142.8, seed 123 reward 1247.017; final checkpoint worse than 1M |
| 20260605_031927_seed3407_ppo_humanoid_colab | 2026-06-05 | `7305160` | `configs/ppo_humanoid_colab.json` | 3407 | 5000000 | 4500000 checkpoint | 4 | 1250 | 250 | Colab CPU eval | 980.384 | Best checkpoint sweep result, seeds 0-4, std 144.320, mean length 195.8; candidate submission checkpoint |
| 20260605_031927_seed3407_ppo_humanoid_colab | 2026-06-05 | `f0ca6f5` | `configs/ppo_humanoid_colab.json` | 3407 | 5000000 | 4500000 checkpoint | 4 | 1250 | 250 | Colab CPU eval | 927.009 | Formal 10-seed evaluation, seeds 0-9, std 173.737, mean length 186.5, seed 123 reward 1331.420 |
| 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel | 2026-06-05 | `82896dc` | `configs/ppo_humanoid_rlzoo_parallel.json` | 3407 | 1000000 | 1000000 | 4 | 1250 | 250 | Colab CPU | 886.012 | RL Zoo-style PPO branch, seeds 0-4, std 149.381, mean length 144.4, seed 123 reward 900.899; competitive 1M result, continue to 5M and sweep checkpoints |
| 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel | 2026-06-05 | `e5ee03b` | `configs/ppo_humanoid_rlzoo_parallel.json` | 3407 | 5000000 | 5000000 checkpoint | 4 | 1250 | 250 | Colab CPU eval | 2275.000 | Checkpoint sweep result, seeds 0-4, std 814.137, mean length 245.4; best checkpoint in sweep, requires formal 10-seed evaluation before replacing final candidate |
| 20260605_081121_seed3407_ppo_humanoid_rlzoo_parallel | 2026-06-05 | `3596d98` | `configs/ppo_humanoid_rlzoo_parallel.json` | 3407 | 5000000 | 5000000 checkpoint | 4 | 1250 | 250 | Colab CPU eval | 2179.016 | Formal 10-seed evaluation, seeds 0-9, std 666.613, mean length 237.8, seed 123 reward 2596.866; replaces previous final candidate |
| 20260605_153005_seed3407_sac_humanoid_cpu_probe | 2026-06-05 | `31bf771` | `configs/sac_humanoid_cpu_probe.json` | 3407 | 200000 | 200000 | 1 | N/A | 256 | Colab CPU | 896.367 | SAC CPU probe, seeds 0-4, std 302.616, mean length 178.6, seed 123 reward 952.200; promising but far below current PPO final candidate |
| 20260606_035618_seed3407_sac_humanoid_fast_probe | 2026-06-06 | `65a1694` | `configs/sac_humanoid_fast_probe.json` | 3407 | 200000 | 200000 | 1 | N/A | 256 | Kaggle T4 GPU | 481.650 | SAC fast probe, seeds 0-4, std 156.369, mean length 99.2; low score, do not continue this fast-probe branch |
| kaggle_sac_cpu_probe_200k_seed3407 | 2026-06-06 | `65a1694` | `configs/sac_humanoid_cpu_probe.json` | 3407 | 200000 | 200000 | 1 | N/A | 256 | Kaggle T4 GPU | 528.677 | Same SAC cpu_probe config rerun on Kaggle, seeds 0-4, std 41.109, mean length 103.2, seed 123 reward 533.753; does not reproduce the stronger Colab CPU probe result |
| colab_sac_cpu_probe_200k_seed3407 | 2026-06-06 | `65a1694` | `configs/sac_humanoid_cpu_probe.json` | 3407 | 200000 | 200000 | 1 | N/A | 256 | Colab CPU | 579.108 | Same SAC cpu_probe config rerun on Colab, seeds 0-4, std 55.303, mean length 113.4, seed 123 reward 620.384; below continuation threshold |
| colab_sac_cpu_probe_200k_seed3407 | 2026-06-06 | `efd8770` | `configs/sac_humanoid_cpu_probe.json` | 3407 | 1000000 | 1000000 | 1 | N/A | 256 | Colab CPU | 1520.323 | SAC continued to 1M, seeds 0-4, std 909.567, mean length 282.0, seed 123 reward 2816.869; improved but still below PPO final candidate on mean reward |
| colab_sac_cpu_probe_200k_seed3407 | 2026-06-06 | `efd8770` | `configs/sac_humanoid_cpu_probe.json` | 3407 | 1000000 | 900000 checkpoint | 1 | N/A | 256 | Colab CPU eval | 5379.517 | SAC 900k checkpoint formal 10-seed evaluation, std 129.155, mean length 993.6, seed 123 reward 4472.476; replaces PPO final candidate |

## 记录规范

- 每次改算法参数前先提交 Git，再训练。
- Run ID 使用 `runs/` 下的目录名。
- `Actual Steps` 以 `runs/<run_id>/metadata.json` 的 `last_num_timesteps` 为准。
- `Mean Raw Reward` 使用 `evaluate.py` 输出的 `mean_reward`，不要使用训练中的 normalized reward。
- 备注里记录是否断点续训、Colab 是否中断、是否出现依赖或环境异常。
