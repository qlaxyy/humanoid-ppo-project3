# 训练视频与复现命令说明

## 1. 视频结构

视频播放内容及其时间轴：

1. 训练开始阶段：00:00--03:03
2. 训练中期阶段：03:03--07:02
3. 训练尾声截图：07:02--07:17
4. 初筛过程：07:17--08:28
5. 最终测试结果：08:28--08:43
6. 行走效果：08:43--09:34

## 2. 训练录屏命令

下面四段命令使用同一个 run 名称：`local_sac_cpu_5m_seed3407`。按顺序运行即可复现完整训练流程。

### 2.1 0 到 100,000 步：多输出模式

用于录制训练开始阶段。

```bat
python train_sac.py --config configs/sac_humanoid_cpu_probe.json --target-steps 100000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000 --run-name local_sac_cpu_5m_seed3407
```

### 2.2 100,000 到 2,900,000 步：安静输出模式

用于长时间训练，不作为重点录屏片段。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 100000 --target-steps 2900000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

### 2.3 2,900,000 到 3,000,000 步：多输出模式

用于录制训练中期阶段。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 2900000 --target-steps 3000000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000
```

### 2.4 3,000,000 到 5,000,000 步：安静输出模式

用于完成最终训练并保存最终 policy。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 3000000 --target-steps 5000000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

## 3. 训练后测试录屏命令

训练完成后，测试流程分为 checkpoint 初筛、正式 10-seed 测试和固定 seed 单次测试。

### 3.1 Checkpoint 初筛

扫描保存的 checkpoint，选择 mean reward 最高的 checkpoint 作为最终候选。

```bat
python evaluate_checkpoints.py --run-dir runs/local_sac_cpu_5m_seed3407 --every 100000 --seeds 0 1 2 3 4 --episodes-per-seed 1 --device cpu
```

本次初筛中 `5000000` 步 checkpoint 排名第一，因此作为最终 policy。

### 3.2 正式 10-seed 测试

```bat
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
```

结果：

```text
mean_reward=6780.676
std_reward=24.244
mean_length=1000.0
```

### 3.3 固定 seed=8 单次测试

```bat
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 8 --episodes 1 --device cpu
```

结果：

```text
raw_reward=6817.721
length=1000
```

## 4. 最终 policy 行走视频生成命令

在本地 Windows 环境中使用 `glfw` 后端：

```bat
python record_video.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 3407 --episodes 1 --device cpu --backend glfw --fps 20
```

视频保存目录：

```text
runs/local_sac_cpu_5m_seed3407/videos/
```
