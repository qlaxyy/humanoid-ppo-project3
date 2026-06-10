# 人工智能基础及应用大作业 3 报告

## 1. 任务背景

本次作业要求在 Gymnasium MuJoCo `Humanoid-v5` 环境中训练强化学习智能体，使三维人形机器人在保持平衡的同时尽可能向前运动。该任务属于高维连续控制问题：智能体需要根据身体姿态、速度、关节受力等观测信息，输出连续动作来控制多个关节的力矩。

`Humanoid-v5` 的动作空间为 17 维连续动作，观测空间默认约为 348 维。环境原始奖励主要由存活奖励、向前速度奖励、控制代价和接触代价组成。最终评测成绩以原生环境 `step()` 返回的原始奖励累计值为准，训练过程中可以使用不同算法和训练技巧，但最终评估不使用归一化后的奖励作为成绩。

本任务的难点主要包括：状态维度高、动作维度高、机器人容易摔倒、训练结果对随机种子和训练过程较敏感，以及长时间训练容易受到平台中断影响。

## 2. 环境与依赖

实验环境固定为：

```text
env_id: Humanoid-v5
gymnasium==1.2.3
mujoco==3.8.1
stable-baselines3==2.8.0
torch==2.7.1
numpy==2.2.6
```

完整依赖记录在 `requirements.txt` 中。最终模型在本地 Windows + Conda CPU 环境中完成训练：

```text
Python: 3.12.13
Platform: Windows 11
CUDA: not used
training seed: 3407
```

本项目未修改 Gymnasium/MuJoCo 源码，也没有修改环境物理参数、奖励函数或 `step()` 返回值。

## 3. 方法

本项目先后尝试了 PPO、RL Zoo 风格 PPO、SAC 和 TD3 探针实验。最终采用 Soft Actor-Critic（SAC）算法。SAC 是一种最大熵 off-policy actor-critic 方法，适用于连续动作控制任务。它在优化累计奖励的同时鼓励策略保持一定随机性，从而提升探索能力和训练稳定性。

最终 SAC 配置如下：

```text
algorithm: SAC
policy: MlpPolicy
learning_rate: 3e-4
buffer_size: 300000
learning_starts: 5000
batch_size: 256
gamma: 0.99
tau: 0.005
train_freq: 4
gradient_steps: 1
ent_coef: auto
policy_net_arch: [256, 256]
qf_net_arch: [256, 256]
normalize_observation: false
normalize_reward: false
```

最终训练 run 为：

```text
run_id: local_sac_cpu_5m_seed3407
target_steps: 5,000,000
selected_checkpoint: 5,000,000 steps
```

SAC 最终模型没有使用 VecNormalize，因此提交和测试时不需要额外加载归一化统计文件。

## 4. 可复现性设计

为保证实验可复现，项目中采取了以下措施：

- 固定依赖版本，并在 `requirements.txt` 中记录。
- 固定环境为 `Humanoid-v5`。
- 固定训练随机种子 `seed=3407`。
- 测试脚本支持命令行传入 seed，并通过 `env.reset(seed=...)` 控制评估初始状态。
- 训练步数不超过作业规定的 `5,000,000` 环境交互步。
- 每隔固定步数保存 checkpoint，便于断点恢复和回溯选择最佳模型。
- 使用 `evaluate.py` 输出多 seed 原始累计奖励，避免只依赖单个 seed。
- 最终模型直接提交 policy checkpoint 文件，便于助教加载测试。

需要说明的是，深度强化学习训练过程存在一定非确定性，尤其是 off-policy 算法、神经网络初始化、经验回放采样和硬件平台差异都可能造成训练轨迹差异。因此本项目最终以提交的 policy checkpoint 在原生环境中的 raw reward 评估结果为准。

## 5. 实验过程

### 5.1 PPO 基线

最初使用 PPO 作为基线算法。PPO 能够跑通完整训练、保存和评估流程，但在 Humanoid-v5 上分数较低。普通 PPO 100 万步评估约为 833；继续训练到 500 万步后，最终模型反而退化，checkpoint sweep 中较好的 PPO 模型也明显低于后续 SAC。

### 5.2 RL Zoo 风格 PPO

随后参考 RL Baselines3 Zoo 风格的 Humanoid PPO 参数进行调参。该分支比普通 PPO 明显更强，500 万步 checkpoint 的正式 10-seed 平均原始奖励为：

```text
mean_reward: 2179.016
```

该结果曾作为中间候选模型，但仍未达到稳定走满 1000 步的水平。

### 5.3 SAC 实验

SAC 初期结果波动较大，早期 20 万步实验在不同平台上表现差异明显。继续训练后，SAC 在约 90 万步附近开始出现稳定走满 1000 步的 checkpoint。后续训练到 500 万步并进行 checkpoint sweep，最终本地 CPU 训练得到的 500 万步 checkpoint 表现最好。

本地 500 万步 checkpoint sweep 的前几名如下：

```text
step       mean_reward   std_reward   mean_length
5000000   6784.349      21.355       1000.0
4600000   6744.175      71.296       1000.0
4900000   6702.615      22.991       1000.0
4200000   6652.197      16.529       1000.0
4300000   6642.718      21.127       1000.0
```

最终选择 `5000000` 步 checkpoint 作为提交模型。

### 5.4 TD3 探针

本项目还尝试了 TD3 fast probe。TD3 是经典的连续控制 off-policy 方法，但在本机 CPU 上早期训练速度低于 SAC，且没有表现出比 SAC 更有利的趋势，因此未作为最终方案。

## 6. 最终结果

最终 policy 文件为：

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
```

正式 10-seed 原始累计奖励评估命令：

```bash
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
```

评估结果：

```text
episodes: 10
mean_reward: 6780.676
std_reward: 24.244
min_reward: 6753.711
max_reward: 6817.721
mean_length: 1000.0
```

各 seed 结果：

```text
[6809.433, 6802.217, 6791.833, 6757.746, 6760.515,
 6754.391, 6760.397, 6798.800, 6817.721, 6753.711]
```

固定 `seed=123` 的测试命令：

```bash
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

测试结果：

```text
raw_reward: 6780.809
length: 1000
terminated: False
truncated: True
```

这说明最终模型能够稳定运行到环境默认最大 episode 长度，并获得较高的原始累计奖励。

## 7. 分析与讨论

从实验结果看，SAC 明显优于本项目中尝试的 PPO 和 TD3 分支。PPO 在 Humanoid-v5 上虽然训练稳定、速度较快，但较难获得高分；SAC 在早期波动较大，但一旦学会稳定步态后，收益提升明显。

训练过程中观察到一个重要现象：强化学习 checkpoint 表现并不一定随训练步数单调提升。早期 SAC 在 90 万步、190 万步和 400 万步均出现过强 checkpoint，最终本地训练的 500 万步 checkpoint 达到当前最高分。因此本项目采用 checkpoint sweep 方式选择最终模型，而不是简单使用 `latest_model.zip`。

本地 CPU 训练虽然耗时较长，但避免了 Colab/Kaggle 断连问题，也更方便录制训练过程。最终模型在 10 个不同测试 seed 上均达到 1000 步，说明策略已经较稳定，不只是对单个 seed 过拟合。

## 8. 提交材料说明

建议提交以下文件和材料：

- 完整源代码。
- `requirements.txt`。
- 最终 policy 文件：
  `runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip`
- 最终评估结果：
  `runs/local_sac_cpu_5m_seed3407/evaluations/raw_eval_20260608_101517.json`
  和对应 CSV。
- 报告 PDF。
- 训练过程分段录屏，总时长约 10 分钟。
- 最终结果截图，包括训练结束截图、`evaluate.py` 输出截图、`test.py` 输出截图。
- 可选：最终策略行走视频。

## 9. 参考资料

- Gymnasium Humanoid-v5 documentation: https://gymnasium.farama.org/environments/mujoco/humanoid/
- Soft Actor-Critic: https://arxiv.org/abs/1801.01290
- Soft Actor-Critic Algorithms and Applications: https://arxiv.org/abs/1812.05905
- Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- Twin Delayed DDPG: https://arxiv.org/abs/1802.09477
- Stable-Baselines3 documentation: https://stable-baselines3.readthedocs.io/
