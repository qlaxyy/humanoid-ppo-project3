# Reproducibility Guide

## 固定内容

- 环境：`Humanoid-v5`
- 依赖：见根目录 `requirements.txt`
- 随机种子：默认 `3407`，可通过 `--seed` 覆盖
- 最大环境交互步数：`5,000,000`
- 默认算法：PPO
- 默认归一化：observation 和 reward 都在训练时归一化

## 随机性来源

本项目在 `humanoid_rl/reproducibility.py` 中固定：

- `PYTHONHASHSEED`
- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- Stable-Baselines3 seed helper
- cuDNN deterministic 设置

环境构造时还会固定：

- `env.reset(seed=...)`
- `env.action_space.seed(...)`
- `env.observation_space.seed(...)`
- 每个并行环境使用 `seed + rank`

注意：跨 CPU/GPU、不同 CUDA/cuDNN、不同操作系统时，深度强化学习很难保证逐 bit 完全一致。作业复现的重点是版本、seed、配置、步数和评估流程一致。

## 训练步数

PPO 按 rollout 批量采样。默认配置：

```text
n_envs = 4
n_steps = 1250
rollout_size = 5000
```

因此 `1,000,000` 和 `5,000,000` 都能被整除。`train.py` 会拒绝可能导致 PPO 超步数的配置。

## 模型与归一化

训练后必须同时保存并提交：

```text
runs/<run_id>/models/latest_model.zip
runs/<run_id>/models/vecnormalize_latest.pkl
```

如果只提交模型而不提交归一化统计量，使用 normalized observation 训练得到的策略在测试时会看到错误尺度的观测，结果无效。

## 原始奖励评估

`evaluate.py` 和 `test.py` 不使用 VecNormalize 的 normalized reward。它们直接创建原始 Gymnasium 环境：

```python
obs, info = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action)
```

代码只手动归一化 observation，然后把原始 `reward` 累加为最终成绩。

## 建议复现实验命令

```bash
pip install -r requirements.txt
python smoke_test.py --steps 5
python train.py --config configs/ppo_humanoid_colab.json --target-steps 1000000
python evaluate.py --run-dir runs/<run_id> --seeds 0 1 2 3 4
python train.py --resume-from runs/<run_id> --target-steps 5000000
python evaluate.py --run-dir runs/<run_id> --seeds 0 1 2 3 4 5 6 7 8 9
```

## 提交前检查

- `requirements.txt` 版本没有被改成范围写法
- `config.json` 中 `env_id` 是 `Humanoid-v5`
- `metadata.json` 中 `last_num_timesteps <= 5000000`
- `latest_model.zip` 存在
- `vecnormalize_latest.pkl` 存在
- `test.py --seed <seed>` 可以运行
- 报告中的分数来自 `evaluate.py` 或 `test.py` 的原始 reward

