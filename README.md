# AI Fundamentals Assignment 3: Humanoid-v5

本项目用于完成“人工智能基础及应用大作业 3”：在 Gymnasium MuJoCo `Humanoid-v5` 环境中训练强化学习智能体。

核心策略：

- 环境固定为 `Humanoid-v5`
- 依赖版本固定在 `requirements.txt`
- 算法使用 Stable-Baselines3 的 PPO
- 训练默认使用 `VecNormalize`，并保存归一化统计量
- 测试和评估脚本使用原始 Gymnasium reward，不使用归一化 reward 作为最终分数
- 总环境交互步数硬限制为 `5,000,000`

## 目录结构

```text
.
├── configs/ppo_humanoid_colab.json  # 默认 PPO baseline 配置
├── configs/ppo_humanoid_stable.json # 更保守的 PPO 对比配置
├── humanoid_rl/                     # 可复现训练、环境、评估工具
├── train.py                         # 训练与断点续训
├── evaluate.py                      # 多 seed 原始奖励评估
├── test.py                          # 单次测试，支持 --seed
├── smoke_test.py                    # 环境和版本快速检查
├── EXPERIMENT_LOG.md                # 实验记录表
├── ASSIGNMENT_NOTES.md              # 作业规则摘要
└── docs/                            # 复现说明和报告提纲
```

训练产物默认保存在 `runs/<run_name>/`，其中包括模型、归一化参数、TensorBoard 日志、评估结果和元数据。

## 安装

建议使用 Python 3.11。Colab 通常已经提供 Python、CUDA 和 PyTorch 环境，但为了满足作业版本要求，仍建议先安装本项目依赖。

```bash
pip install -r requirements.txt
```

检查环境：

```bash
python smoke_test.py --steps 5
```

如果只是在本机没有完整依赖的情况下看流程，可以临时放宽版本检查：

```bash
python smoke_test.py --steps 5 --no-strict-versions
```

正式训练和提交前不要放宽版本。

## Colab 训练

推荐工作流：

1. 在本机 VS Code 修改代码，并 push 到 GitHub。
2. 在 Colab 使用 GPU runtime。
3. 每次新 runtime 把代码 clone 到 `/content`，这是临时盘但速度快。
4. 把训练产物保存到 Google Drive，例如 `/content/drive/MyDrive/humanoid_runs`。
5. Colab 断线后重新 clone 代码、安装依赖，再从 Drive 中的 run 目录恢复。

Colab 中常用命令。每次新 runtime 先准备代码和依赖：

```bash
git clone https://github.com/qlaxyy/humanoid-ppo-project3.git /content/humanoid-ppo-project3-code
cd /content/humanoid-ppo-project3-code
pip install -r requirements.txt
python smoke_test.py --steps 5
```

第一次先跑 baseline，并把输出保存到 Drive：

```bash
python train.py --config configs/ppo_humanoid_colab.json --target-steps 1000000 --device auto --output-dir /content/drive/MyDrive/humanoid_runs
```

如果 baseline 日志里 `approx_kl` 和 `clip_fraction` 偏高，可以再跑一个更保守的 PPO 对比实验：

```bash
python train.py --config configs/ppo_humanoid_stable.json --target-steps 1000000 --device auto --output-dir /content/drive/MyDrive/humanoid_runs
```

也可以尝试 RL Baselines3 Zoo 风格的 Humanoid PPO 参数。先短跑 100 万步对比，不要直接训满：

```bash
python train.py --config configs/ppo_humanoid_rlzoo_parallel.json --target-steps 1000000
```

该配置基于 RL Baselines3 Zoo 中 `Humanoid-v4` 的 tuned PPO 思路，并适配为 4 个并行环境以保持 rollout 与作业步数对齐。

继续训练到作业允许的上限：

```bash
python train.py --resume-from runs/<run_name> --target-steps 5000000 --device auto
```

如果训练中途断线，`train.py --resume-from` 会优先加载 `latest_model.zip`；如果训练没有正常结束，则自动寻找 `models/checkpoint_model_*_steps.zip` 和对应的 VecNormalize 统计量。

注意：默认配置的 `n_envs=4`、`n_steps=1250`，每个 PPO rollout 是 `5000` 步，能整除 `1,000,000` 和 `5,000,000`，避免 PPO 自动向上采样导致超过作业步数限制。

## 评估和测试

多随机种子评估，结果会写入 `runs/<run_name>/evaluations/`：

```bash
python evaluate.py --run-dir /content/drive/MyDrive/humanoid_runs/<run_name> --seeds 0 1 2 3 4 --episodes-per-seed 1
```

单次测试，显式把命令行 seed 传给 `env.reset(seed=...)`：

```bash
python test.py --run-dir /content/drive/MyDrive/humanoid_runs/<run_name> --seed 123 --episodes 1
```

汇总所有已经评估过的实验：

```bash
python summarize_experiments.py --runs-dir /content/drive/MyDrive/humanoid_runs
```

如果训练到 5,000,000 步后 `latest_model.zip` 退化，可以评估中间 checkpoint 并选择分数最高的合法模型：

```bash
python evaluate_checkpoints.py --run-dir runs/<run_name> --every 500000 --seeds 0 1 2 3 4
```

更精细地评估指定 checkpoint：

```bash
python evaluate_checkpoints.py --run-dir runs/<run_name> --steps 1000000 1500000 2000000 --seeds 0 1 2 3 4
```

确定最佳 checkpoint 后，可以直接用步数评估或测试：

```bash
python evaluate.py --run-dir runs/<run_name> --checkpoint-step 4500000 --seeds 0 1 2 3 4 5 6 7 8 9
python test.py --run-dir runs/<run_name> --checkpoint-step 4500000 --seed 123 --episodes 1
```

如果需要可视化：

```bash
python test.py --run-dir runs/<run_name> --seed 123 --render-mode human
```

## 实验记录

每次训练开始前，建议在 `EXPERIMENT_LOG.md` 增加一行，记录：

- Git commit
- 配置文件
- seed
- 目标步数
- n_envs / n_steps / batch_size
- 使用硬件
- 最终平均分
- 备注

脚本也会自动写入：

- `runs/<run_name>/config.json`
- `runs/<run_name>/metadata.json`
- `runs/<run_name>/evaluations/*.json`
- `runs/<run_name>/evaluations/*.csv`

## GitHub 同步

本地修改代码后：

```bash
git status
git add <changed-files>
git commit -m "Describe the change"
git push
```

更多解释见 `docs/git_workflow.md`。

## 作业合规重点

- 不修改 MuJoCo 原生物理参数。
- 不修改环境 `step()` 返回值作为最终成绩。
- 训练最多 `5,000,000` 环境交互步。
- 使用归一化时，模型和 `vecnormalize_latest.pkl` 必须一起提交。
- `test.py` 支持通过命令行设置 seed。
- 最终报告分数应来自原始环境累计 reward。
