# Report Outline

最终报告建议按下面结构写。

## 1. 问题背景

- 介绍 Humanoid-v5 环境
- 说明智能体、状态、动作、奖励、episode 终止条件
- 说明本任务难点：高维连续控制、容易摔倒、训练不稳定

## 2. 环境与依赖

- Python 版本
- `gymnasium==1.2.3`
- `mujoco==3.8.1`
- `stable-baselines3==2.8.0`
- `torch==2.7.1`
- 训练硬件：本机 CPU / Colab GPU
- 随机种子

## 3. 方法

- 算法：PPO
- 策略网络和值函数网络结构
- 关键超参数：learning rate、gamma、gae_lambda、clip_range、n_steps、batch_size、n_envs
- 观测和奖励归一化方法
- 断点保存和恢复机制

## 4. 可复现性设计

- 版本固定
- 随机种子固定
- 不修改环境物理参数
- 训练步数限制
- 保存模型与 VecNormalize 统计量
- 原始奖励评估脚本

## 5. 实验结果

- 不同步数的平均原始 reward
- 多 seed 评估表
- 最终分数截图
- 训练曲线截图，可来自 TensorBoard
- 是否观察到机器人学会稳定前进

## 6. 分析与讨论

- 训练是否稳定
- 哪些参数影响明显
- 没有 GPU 或 Colab 中断对实验的影响
- 当前方法的不足和可能改进

## 7. 最终得分

- 给出最终使用模型路径
- 给出评估命令
- 给出最终 raw reward 分数

