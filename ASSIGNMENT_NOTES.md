# Assignment Notes

来源：工作区 PDF `人工智能基础及应用大作业3说明（修改）.pdf` 与官方页面 <https://gymnasium.farama.org/environments/mujoco/humanoid/>。

## 任务

在 Gymnasium MuJoCo `Humanoid-v5` 环境中训练智能体，使三维人形机器人尽可能快地向前行走，同时保持平衡不摔倒。

## 环境规则

- 必须使用 `Humanoid-v5`
- `gymnasium==1.2.3`
- `mujoco==3.8.1`
- 使用 PyTorch、Stable-Baselines3 等第三方库时，必须在 `requirements.txt` 中标注具体版本

## 代码限制

- 必须固定随机种子，覆盖环境初始化、模型初始化、采样等随机过程
- 测试脚本必须支持命令行传参设置 `env.reset(seed=...)`
- 严禁修改重力、摩擦力、刚体质量、关节力矩范围等原生物理参数
- 严禁修改原生 `step()` 返回值作为最终测试得分
- 训练过程可以用奖励塑形或归一化，但最终评测成绩必须是原生环境输出的原始累计奖励
- 与环境交互步数不得超过 `5,000,000`
- 如果训练使用 `NormalizeObservation`、`NormalizeReward` 或等价归一化方法，必须保存均值、方差等统计参数，并在测试时正确加载

## Humanoid-v5 关键信息

- 动作空间：17 维连续动作，用于控制人形机器人多个关节
- 默认观测空间：348 维，包括身体姿态、速度、质心相关信息、关节力和接触力等
- 奖励主要由存活奖励、向前速度奖励、控制代价、接触代价组成
- 默认 episode 最多 1000 步
- 躯干高度不在健康范围时通常会终止

## 本项目如何对应作业要求

- `train.py` 固定 `Humanoid-v5`，配置校验禁止其他环境
- `requirements.txt` 固定核心依赖版本
- `humanoid_rl/reproducibility.py` 统一固定 Python、NumPy、PyTorch、CUDA、SB3 随机种子
- `humanoid_rl/envs.py` 不传入自定义物理参数，不改环境 reward 或 step
- `train.py` 将训练上限限制在 `5,000,000`，并检查 PPO rollout 步数是否会溢出
- `train.py` 保存 `latest_model.zip` 和 `vecnormalize_latest.pkl`
- `test.py` 明确通过 `--seed` 调用 `env.reset(seed=...)`
- `evaluate.py` 和 `test.py` 手动加载观测归一化统计量，但累计的是原始环境 reward

