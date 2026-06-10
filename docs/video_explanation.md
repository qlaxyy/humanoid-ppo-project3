# 训练视频与命令说明

本次最终训练目标为 5,000,000 环境交互步。由于完整训练耗时较长，视频按照作业补充通知采用分段方式展示训练开始、中期和结尾阶段，并补充最终评估截图。

最终 policy 文件：

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

## 输出模式

- 多输出模式：用于录屏展示。开启进度条和指标表，较高频打印 reward、episode length、fps 和训练损失等指标。
- 安静输出模式：用于长时间训练。关闭进度条，只低频输出进度，避免终端输出过多。

## 分段训练命令

下面四段命令使用同一个 run 名称：`local_sac_cpu_5m_seed3407`。按顺序运行即可复现完整的 5,000,000 步训练过程。

### 1. 0 到 100,000 步：多输出模式

用于展示训练开始阶段。

```bat
python train_sac.py --config configs/sac_humanoid_cpu_probe.json --target-steps 100000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000 --run-name local_sac_cpu_5m_seed3407
```

### 2. 100,000 到 2,900,000 步：安静输出模式

用于完成较长的中前期训练。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 100000 --target-steps 2900000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

### 3. 2,900,000 到 3,000,000 步：多输出模式

用于展示训练中期阶段。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 2900000 --target-steps 3000000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000
```

### 4. 3,000,000 到 5,000,000 步：安静输出模式

用于完成最终训练并保存最终 policy。

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
