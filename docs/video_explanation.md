# 训练视频与命令说明

本次最终训练目标为 5,000,000 环境交互步。由于完整训练耗时较长，视频按照作业补充通知采用分段录制方式：训练开始片段、训练中期片段、训练尾声截图和最终测试结果。

最终提交 policy 文件保持不变：

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
```

正式 10-seed 评估结果：

```text
mean_reward=6780.676
std_reward=24.244
mean_length=1000.0
```

固定 seed=123 单次测试结果：

```text
raw_reward=6780.809
length=1000
```

## 输出模式说明

- 多输出模式：用于录屏展示。开启进度条和 `--metric-table`，并较高频打印指标表，便于观察 reward、episode length、fps 和训练损失等动态变化。
- 安静输出模式：用于长时间训练。关闭进度条，不打印指标表，只低频输出进度，避免终端输出过多。

`--metric-table` 会使用普通小数格式打印 reward，例如：

```text
ep_rew_mean  4432.123
```

避免 Stable-Baselines3 原生日志中可能出现的科学计数法，例如 `4.4e+03`。

## 分段命令

### 1. 0 到 100,000 步：多输出录屏

用途：录制训练开始阶段，展示训练从零启动、reward 逐步变化、进度条和指标表。

说明：这段使用独立演示 run，不替代最终提交模型。

```bat
python train_sac.py --config configs/sac_humanoid_cpu_probe.json --target-steps 100000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000 --run-name local_sac_start_demo_0k_to_100k
```

如果该 demo 文件夹已经存在，可以把 `--run-name` 改成新的名字，例如：

```bat
--run-name local_sac_start_demo_0k_to_100k_v2
```

### 2. 100,000 到 2,900,000 步：安静输出

用途：长时间正式训练，不录屏或只保留少量过程输出。

说明：这是最终训练 run 的中间阶段。若从 100,000 步继续正式训练到 2,900,000 步，使用：

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 100000 --target-steps 2900000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

### 3. 2,900,000 到 3,000,000 步：多输出录屏

用途：录制训练中期阶段，展示模型已经学到较好策略后，reward、episode length、评估结果和 checkpoint 的动态变化。

先创建独立中期演示 run：

```bat
python prepare_mid_training_demo.py --source-run runs/local_sac_cpu_5m_seed3407 --source-step 2900000 --demo-run runs/local_sac_mid_demo_2900k_to_3000k --target-steps 3000000
```

然后录制中期继续训练片段：

```bat
python train_sac.py --resume-from runs/local_sac_mid_demo_2900k_to_3000k --resume-step 2900000 --target-steps 3000000 --device cpu --quiet --progress-bar --status-freq 2000 --metric-table --checkpoint-freq 50000 --eval-freq 50000
```

如果希望指标表更频繁，可以把 `--status-freq 2000` 改为：

```bat
--status-freq 1000
```

### 4. 3,000,000 到 5,000,000 步：安静输出

用途：完成正式 5,000,000 步训练，主要用于得到最终 policy。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 3000000 --target-steps 5000000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

## 最终测试命令

10-seed 原始累计奖励评估：

```bat
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
```

固定 seed=123 单次测试：

```bat
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```
