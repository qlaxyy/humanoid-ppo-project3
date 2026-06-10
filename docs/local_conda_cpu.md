# Local Conda CPU Workflow

This workflow is for running a short local CPU training/evaluation session on
Windows. It is useful for screen recording because it does not depend on Colab
or network stability.

## Create Environment

Open Anaconda Prompt or VS Code PowerShell in the project directory:

```powershell
cd "E:\Sophomore_S2\AI Fundamentals and Applications\大作业3"
```

Create and activate a clean environment:

```powershell
conda create -n humanoid-rl python=3.12 -y
conda activate humanoid-rl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the install asks for a restart in a notebook, restart that notebook runtime.
In a normal terminal, close and reopen the terminal only if Python imports look
stale.

## Smoke Test

Run:

```powershell
python smoke_test.py --steps 5
```

Expected important lines:

```text
gymnasium==1.2.3
mujoco==3.8.1
Smoke test finished
```

## Local Speed Probe

Run a short local CPU SAC probe. This is not intended to replace the final
submission model; it is for recording and measuring local speed.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python train_sac.py `
  --config configs/sac_humanoid_cpu_probe.json `
  --target-steps 20000 `
  --device cpu `
  --quiet `
  --no-progress-bar `
  --status-freq 2000 `
  --run-name local_sac_cpu_probe_20k
```

If this is too slow, stop it with `Ctrl+C`. The project already has the final
Colab-trained policy, so the local run is only process evidence.

## Resume Local Probe

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python train_sac.py `
  --resume-from runs/local_sac_cpu_probe_20k `
  --target-steps 50000 `
  --device cpu `
  --quiet `
  --no-progress-bar `
  --status-freq 5000
```

## Evaluate Final Colab Policy Locally

Copy the final policy run directory from Google Drive if it is not already in
the local `runs/` folder. The final policy is:

```text
runs/colab_sac_cpu_probe_200k_seed3407/models/checkpoint_model_4000000_steps.zip
```

Then run:

```powershell
python evaluate.py `
  --run-dir runs/colab_sac_cpu_probe_200k_seed3407 `
  --checkpoint-step 4000000 `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --episodes-per-seed 1 `
  --device cpu

python test.py `
  --run-dir runs/colab_sac_cpu_probe_200k_seed3407 `
  --checkpoint-step 4000000 `
  --seed 123 `
  --episodes 1 `
  --device cpu
```

## Screen Recording Checklist

Record these items:

- Environment version check with `smoke_test.py`.
- Local training command starting and printing progress lines.
- Existing final policy file in `runs/colab_sac_cpu_probe_200k_seed3407/models/`.
- Final evaluation commands and raw reward output.
- Optional rendered Humanoid video generated from the final policy.

