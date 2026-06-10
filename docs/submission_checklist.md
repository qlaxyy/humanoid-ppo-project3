# Submission Checklist

提交前逐项检查。

## 代码

- `requirements.txt` 存在，且包含具体版本号
- `train.py` 可以启动训练
- `test.py` 可以通过 `--seed` 设置测试 seed
- `evaluate.py` 可以输出多 seed 原始 reward
- 没有修改 Gymnasium/MuJoCo 源码
- 没有在最终评估中使用 normalized reward

## 训练产物

- `runs/<run_id>/models/latest_model.zip`
- `runs/<run_id>/models/vecnormalize_latest.pkl`
- 如果训练被中断，可以先用 `train.py --resume-from <run_dir>` 从最新 checkpoint 恢复，再生成最终 `latest_model.zip`
- `runs/<run_id>/config.json`
- `runs/<run_id>/metadata.json`
- `runs/<run_id>/evaluations/raw_eval_*.json`
- `runs/<run_id>/evaluations/raw_eval_*.csv`

## 步数限制

- `metadata.json` 中 `last_num_timesteps <= 5000000`
- `config.json` 中 `env_id == "Humanoid-v5"`
- 没有额外训练超过上限后又只提交最后模型

## 报告

- 问题背景描述
- 环境和依赖版本
- 方法描述：PPO、网络结构、关键超参数
- 可复现性说明：seed、版本、步数、归一化参数
- 实验结果：训练曲线、多 seed 评估、最终分数截图
- 最终得分和对应模型路径

## 截图和录屏

- 最后得到的分数截图，以“学号-姓名-分数”命名
- 训练开始运行的 5 分钟录屏
- 包含最终结果的 5 分钟录屏
- 文件名按作业要求命名

## 压缩包

- 命名：`学号-姓名-大作业3`
- 内容包含报告、完整源代码、模型及归一化参数、分数截图、录屏材料
