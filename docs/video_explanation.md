# Video Explanation

This note can be submitted with the segmented recording to explain why the video
contains separate clips instead of one continuous training session.

## Suggested Caption

本次训练总步数为 5,000,000 环境交互步，完整训练耗时约 10 小时。根据作业补充通知，视频采用分段录制方式展示训练开始、中期和结尾结果，并补充清晰截图说明最终输出。

视频结构如下：

1. `0:00-1:28`：训练开始阶段录屏。
   展示本地 Conda 环境、训练命令启动、SAC 训练开始后的日志输出和进度信息。

2. `1:28-待补充`：训练中期阶段录屏。
   为了补录训练中期画面，使用 `2,500,000` 步 checkpoint 创建了一个独立的演示 run，并从 `2,500,000` 步继续训练到 `2,600,000` 步。该片段用于展示训练中期的 reward、episode length、progress、checkpoint 等动态变化。

3. `待补充-待补充`：训练尾声阶段截图展示。
   由于完整 5,000,000 步训练耗时较长，尾声阶段安排在夜间运行，未能实时录制完整尾声视频。因此视频中展示训练结束后的终端截图，包括 `progress 5000000/5000000`、`Eval num_timesteps=5000000`、模型保存路径和最终评估结果。

4. `最后片段`：最终 policy 测试与结果展示。
   运行 `evaluate.py` 和 `test.py` 加载最终 policy 文件 `checkpoint_model_5000000_steps.zip`，展示原生环境 raw reward。正式 10-seed 评估结果为 `mean_reward=6780.676`，固定 `seed=123` 的 raw reward 为 `6780.809`。

## Output Modes

训练脚本支持两种输出模式：

- 多输出模式：用于录屏展示，会显示较多训练指标、progress 和评估信息，便于观察 reward 的动态变化。
- 安静输出模式：用于长时间训练，只低频输出进度，避免终端输出过多导致后续命令和结果难以回看。

视频中展示的是多输出模式；非录屏时间主要使用安静输出模式完成长时间训练。

## Commands Used For The Middle Clip

Create an isolated middle-stage demo run:

```bash
python prepare_mid_training_demo.py --source-run runs/local_sac_cpu_5m_seed3407 --source-step 2500000 --demo-run runs/local_sac_mid_demo_2500k_to_2600k --target-steps 2600000
```

Record the middle-stage continuation:

```bash
python train_sac.py --resume-from runs/local_sac_mid_demo_2500k_to_2600k --resume-step 2500000 --target-steps 2600000 --device cpu --quiet --no-progress-bar --status-freq 5000 --checkpoint-freq 50000 --eval-freq 50000
```

The progress lines include full decimal reward values, for example
`recent_ep_rew_mean=4432.123`, so the recording does not rely on Stable-Baselines3
tables that may use scientific notation.

This middle-stage demo does not replace the final submitted policy. The final
submitted policy remains:

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
```
