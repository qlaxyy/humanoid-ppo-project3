# Humanoid-v5 Continuous Control with SAC

本项目完成 Gymnasium MuJoCo `Humanoid-v5` 连续控制任务。仓库创建初期以 PPO 为基线，随后对 RL Zoo 风格 PPO、TD3 和 SAC 进行了对比，最终选择 Stable-Baselines3 的 Soft Actor-Critic（SAC）作为提交方案。

## 最终结果

- 环境：`Humanoid-v5`
- 算法：SAC（`MlpPolicy`）
- 训练步数：`5,000,000`
- 训练 seed：`3407`
- 正式评估：10 个测试 seed，每个 seed 运行 1 个 episode
- 平均原始累计奖励：`6780.676 ± 24.244`
- 最低 / 最高奖励：`6753.711 / 6817.721`
- 平均 episode length：`1000.0`
- 最终 checkpoint：`checkpoint_model_5000000_steps.zip`

正式成绩以多 seed 平均值为主，`6817.721` 是 seed 8 对应的最高单回合结果。

![Checkpoint sweep](docs/checkpoint_sweep_curve.png)

## 方法概述

最终 SAC 配置使用两个隐藏层为 `[256, 256]` 的策略网络和 Q 网络，经验回放池容量为 `300000`，学习率为 `3e-4`，折扣因子为 `0.99`，目标网络软更新系数为 `0.005`。熵系数和目标熵均由算法自动调整。

训练直接使用原生 `Humanoid-v5` 奖励，没有修改环境物理参数，也没有启用观测或奖励归一化。最终评价由原生环境 `step()` 返回的累计 reward 计算。

完整配置见 [`configs/sac_humanoid_cpu_probe.json`](configs/sac_humanoid_cpu_probe.json)。算法选择和实验过程见 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) 与 [`report_final.md`](report_final.md)。

## 项目结构

```text
.
├── configs/
│   ├── sac_humanoid_cpu_probe.json   # 最终 SAC 配置
│   └── ppo_*.json / td3_*.json       # 对比实验配置
├── humanoid_rl/                      # 环境、回调、评估和复现工具
├── docs/
│   ├── checkpoint_sweep_curve.png    # checkpoint 评估曲线
│   ├── checkpoint_sweep_data.csv     # 绘图数据
│   ├── final_candidate.md            # 最终候选模型记录
│   └── video_explanation.md          # 分段训练及视频说明
├── train_sac.py                      # 最终 SAC 训练与断点续训
├── evaluate_checkpoints.py           # checkpoint 初筛
├── evaluate.py                       # 多 seed 原始奖励评估
├── test.py                           # 单 seed 测试
├── record_video.py                   # 直接录制策略视频
├── export_trajectory.py              # 导出动作轨迹
├── render_trajectory.py              # 离线渲染轨迹
├── requirements.txt                  # 固定依赖版本
└── EXPERIMENT_LOG.md                 # 实验结果记录
```

`train.py` 和 `train_td3.py` 分别保留了 PPO、TD3 对比实验流程，不是最终训练入口。

## 环境安装

推荐使用 Conda 创建独立环境：

```bash
conda create -n humanoid-rl python=3.12 -y
conda activate humanoid-rl
pip install -r requirements.txt
```

快速检查 MuJoCo 环境：

```bash
python smoke_test.py --steps 5
```

最终实验使用的关键版本：

```text
gymnasium==1.2.3
mujoco==3.8.1
stable-baselines3==2.8.0
torch==2.7.1
numpy==2.2.6
```

## 训练

一次性运行到作业允许的 500 万步：

```bash
python train_sac.py --config configs/sac_humanoid_cpu_probe.json --target-steps 5000000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000 --run-name local_sac_cpu_5m_seed3407
```

从指定 checkpoint 继续训练：

```bash
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 3000000 --target-steps 5000000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

用于录屏的高频指标模式：

```bash
python train_sac.py --config configs/sac_humanoid_cpu_probe.json --target-steps 100000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000 --run-name local_sac_cpu_5m_seed3407
```

训练脚本会保存配置、运行元数据、checkpoint、评估日志和进度状态。断点续训会恢复模型参数和训练步数，但当前配置不保存 replay buffer，因此续训结果不保证与完全不中断的运行逐位一致。

## 模型选择与评估

先使用 5 个 seed 扫描 checkpoint：

```bash
python evaluate_checkpoints.py --run-dir runs/local_sac_cpu_5m_seed3407 --every 100000 --seeds 0 1 2 3 4 --episodes-per-seed 1 --device cpu
```

对选出的 500 万步 checkpoint 进行正式 10-seed 评估：

```bash
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
```

复现最高单回合结果：

```bash
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 8 --episodes 1 --device cpu
```

评估脚本会将 CSV 和 JSON 结果写入 `runs/<run_id>/evaluations/`。

## 视频生成

Windows 本地使用 GLFW 后端：

```bash
python record_video.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 8 --episodes 1 --device cpu --backend glfw --fps 20
```

如果直接录制遇到 OpenGL 问题，可先导出轨迹再离线渲染：

```bash
python export_trajectory.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 8 --episodes 1 --device cpu
python render_latest_trajectory.py --run-dir runs/local_sac_cpu_5m_seed3407 --backend glfw --fps 20
```

## 模型与生成物

训练模型、运行日志、视频、Word 报告和个人提交压缩包不进入 Git 仓库，避免上传大量二进制文件。最终 policy 已随课程作业单独提交。若需要复现，应按上述命令重新训练，或将模型放回以下位置：

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
```

## 参考资料

- [Gymnasium Humanoid-v5](https://gymnasium.farama.org/environments/mujoco/humanoid/)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
