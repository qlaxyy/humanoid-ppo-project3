# 训练视频与复现命令说明

本次最终训练目标为 5,000,000 环境交互步。由于完整训练耗时较长，视频按作业补充通知采用分段方式展示训练开始、中期、尾声截图、测试过程和最终 policy 行走效果。

最终 policy 文件：

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
```

最终正式评估结果：

```text
10-seed mean_reward=6780.676
seed=3407 raw_reward=6770.456
mean_length=1000.0
```

## 1. 训练录屏命令

下面四段命令使用同一个 run 名称：`local_sac_cpu_5m_seed3407`。按顺序运行即可复现完整训练流程。

### 1.1 0 到 100,000 步：多输出模式

用于录制训练开始阶段。

```bat
python train_sac.py --config configs/sac_humanoid_cpu_probe.json --target-steps 100000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000 --run-name local_sac_cpu_5m_seed3407
```

### 1.2 100,000 到 2,900,000 步：安静输出模式

用于长时间训练，不作为重点录屏片段。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 100000 --target-steps 2900000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

### 1.3 2,900,000 到 3,000,000 步：多输出模式

用于录制训练中期阶段。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 2900000 --target-steps 3000000 --device cpu --quiet --progress-bar --status-freq 1000 --metric-table --checkpoint-freq 50000 --eval-freq 50000
```

### 1.4 3,000,000 到 5,000,000 步：安静输出模式

用于完成最终训练并保存最终 policy。

```bat
python train_sac.py --resume-from runs/local_sac_cpu_5m_seed3407 --resume-step 3000000 --target-steps 5000000 --device cpu --quiet --no-progress-bar --status-freq 100000 --checkpoint-freq 100000 --eval-freq 100000
```

## 2. 训练后测试录屏命令

训练完成后，测试流程分为 checkpoint 初筛、正式 10-seed 测试和固定 seed 单次测试。

### 2.1 Checkpoint 初筛

扫描保存的 checkpoint，选择 mean reward 最高的 checkpoint 作为最终候选。

```bat
python evaluate_checkpoints.py --run-dir runs/local_sac_cpu_5m_seed3407 --every 100000 --seeds 0 1 2 3 4 --episodes-per-seed 1 --device cpu
```

本次初筛中 `5000000` 步 checkpoint 排名第一，因此作为最终 policy。

### 2.2 正式 10-seed 测试

```bat
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
```

结果：

```text
mean_reward=6780.676
std_reward=24.244
mean_length=1000.0
```

### 2.3 固定 seed=3407 单次测试

```bat
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 3407 --episodes 1 --device cpu
```

结果：

```text
raw_reward=6770.456
length=1000
```

## 3. 最终 policy 行走视频生成命令

### 3.1 直接录制 policy 视频

在本地 Windows 环境中使用 `glfw` 后端：

```bat
python record_video.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 3407 --episodes 1 --device cpu --backend glfw --fps 20
```

视频保存目录：

```text
runs/local_sac_cpu_5m_seed3407/videos/
```

### 3.2 备用方式：先导出轨迹再渲染

如果直接录制遇到 MuJoCo/OpenGL 后端问题，使用以下两步：

```bat
python export_trajectory.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 3407 --episodes 1 --device cpu
```

```bat
python render_latest_trajectory.py --run-dir runs/local_sac_cpu_5m_seed3407 --backend glfw --fps 20 --repeat 3
```

渲染结果保存目录：

```text
runs/local_sac_cpu_5m_seed3407/trajectories/videos/
```

## 4. 视频结构建议

视频可以按以下顺序剪辑：

1. 训练开始阶段：展示第 1.1 节命令输出。
2. 训练中期阶段：展示第 1.3 节命令输出。
3. 训练尾声截图：展示训练达到 `5000000/5000000`、保存模型和最终评估结果。
4. 测试过程：展示第 2.1 到 2.3 节命令输出。
5. 行走效果：展示第 3 节生成的 policy 行走视频。
