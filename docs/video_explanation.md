# 训练视频说明

本次训练总步数为 5,000,000 环境交互步，完整训练耗时较长。根据作业补充通知，视频采用分段录制方式展示训练开始、中期和结尾阶段，并补充清晰截图说明最终结果。

## 视频结构

1. `0:00-1:28`：训练开始阶段录屏。展示本地 Conda 环境、训练命令启动、SAC 训练开始后的日志输出和进度信息。
2. `1:28-待补充`：训练中期阶段录屏。为补录训练中期画面，使用 `2,900,000` 步 checkpoint 创建独立演示 run，并从 `2,900,000` 步继续训练到 `3,000,000` 步。该片段用于展示训练中期的 reward、episode length、progress、checkpoint 等动态变化。
3. `待补充-待补充`：训练尾声阶段截图展示。由于完整 5,000,000 步训练耗时较长，尾声阶段安排在夜间运行，未能实时录制完整尾声视频。因此视频中展示训练结束后的终端截图，包括 `progress 5000000/5000000`、`Eval num_timesteps=5000000`、模型保存路径和最终评估结果。
4. 最后片段：最终 policy 测试与结果展示。运行 `evaluate.py` 和 `test.py` 加载最终 policy 文件 `checkpoint_model_5000000_steps.zip`，展示原生环境 raw reward。正式 10-seed 评估结果为 `mean_reward=6780.676`，固定 `seed=123` 的 raw reward 为 `6780.809`。

## 输出模式

训练脚本支持两种输出模式：

- 多输出模式：用于录屏展示，会显示较多训练指标、progress 和评估信息，便于观察 reward 的动态变化。
- 安静输出模式：用于长时间训练，只低频输出进度，避免终端输出过多导致后续命令和结果难以回看。

视频中展示的是多输出模式或较高频 progress 输出；非录屏时间主要使用安静输出模式完成长时间训练。

## 中期片段命令

创建独立的中期演示 run：

```bash
python prepare_mid_training_demo.py --source-run runs/local_sac_cpu_5m_seed3407 --source-step 2900000 --demo-run runs/local_sac_mid_demo_2900k_to_3000k --target-steps 3000000
```

录制中期继续训练片段：

```bash
python train_sac.py --resume-from runs/local_sac_mid_demo_2900k_to_3000k --resume-step 2900000 --target-steps 3000000 --device cpu --quiet --no-progress-bar --status-freq 5000 --metric-table --checkpoint-freq 50000 --eval-freq 50000
```

这里的 progress 行和指标表都会输出普通小数格式的近期 episode 指标，例如：

```text
recent_ep_rew_mean=4432.123 recent_ep_len_mean=1000.0
```

因此录屏不依赖 Stable-Baselines3 表格中的科学计数法显示。

该中期演示 run 只用于补录视频，不替代最终提交的 policy。最终提交 policy 仍为：

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
```
