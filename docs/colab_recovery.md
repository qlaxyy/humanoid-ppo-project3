# Colab Recovery Notes

Colab runtime 是临时虚拟机。断线、重启、空闲超时后：

- `/content` 下的代码和临时文件可能消失
- 已安装的 pip 包通常需要重新安装
- Python 变量会消失
- Google Drive 中的文件会保留

因此本项目采用：

```text
代码：GitHub -> /content/humanoid-ppo-project3-code
训练产物：/content/drive/MyDrive/humanoid_runs
```

## 每次新 runtime 的准备命令

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/qlaxyy/humanoid-ppo-project3.git /content/humanoid-ppo-project3-code
```

如果目录已存在：

```bash
!git -C /content/humanoid-ppo-project3-code pull --ff-only
```

进入代码目录：

```python
%cd /content/humanoid-ppo-project3-code
```

安装依赖和检查环境：

```bash
!pip install -r requirements.txt
!python smoke_test.py --steps 5
```

## 中断后恢复训练

先查看 Drive 中的 run：

```bash
!ls /content/drive/MyDrive/humanoid_runs
```

选择要恢复的目录，例如：

```python
RUN_DIR = '/content/drive/MyDrive/humanoid_runs/20260604_112100_seed3407_ppo_humanoid_colab'
```

恢复到 100 万步：

```bash
!python train.py --resume-from $RUN_DIR --target-steps 1000000 --device auto
```

恢复到 500 万步：

```bash
!python train.py --resume-from $RUN_DIR --target-steps 5000000 --device auto
```

`train.py` 会优先使用：

```text
models/latest_model.zip
models/vecnormalize_latest.pkl
```

如果训练没有正常结束，则自动寻找：

```text
models/checkpoint_model_*_steps.zip
models/checkpoint_model_vecnormalize_*_steps.pkl
models/vecnormalize_*_steps.pkl
```

## 评估

```bash
!python evaluate.py --run-dir $RUN_DIR --seeds 0 1 2 3 4 --episodes-per-seed 1
!python test.py --run-dir $RUN_DIR --seed 123 --episodes 1
```

汇总所有实验：

```bash
!python summarize_experiments.py --runs-dir /content/drive/MyDrive/humanoid_runs
```

