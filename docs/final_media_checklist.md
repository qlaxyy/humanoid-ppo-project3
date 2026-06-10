# Final Media Checklist

## Required Screenshots

Keep clear screenshots of the following:

1. Training completion:
   - Shows `progress 5000000/5000000`.
   - Shows `Eval num_timesteps=5000000`.
   - Shows `Saved model ... runs\local_sac_cpu_5m_seed3407\models`.

2. Final 10-seed evaluation:

```bash
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
```

Expected key result:

```text
mean_reward: 6780.676
std_reward: 24.244
mean_length: 1000.0
```

3. Fixed-seed test:

```bash
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

Expected key result:

```text
seed=123 raw_reward=6780.809 length=1000
```

4. Final policy file:

```bash
dir runs\local_sac_cpu_5m_seed3407\models\checkpoint_model_5000000_steps.zip
```

## Recommended Recording Structure

The supplemental notice says full training recording is not required. A segmented
recording of about 10 minutes is enough.

Recommended video structure:

1. Environment and project, about 1-2 minutes:

```bash
python smoke_test.py --steps 5
dir runs\local_sac_cpu_5m_seed3407\models\checkpoint_model_5000000_steps.zip
```

2. Training process evidence, about 3-4 minutes:
   - Show the terminal section where training reaches 4.8M-5.0M steps.
   - Show `Eval num_timesteps=5000000`.
   - Show checkpoint sweep output.

3. Final evaluation, about 3 minutes:

```bash
python evaluate.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seeds 0 1 2 3 4 5 6 7 8 9 --episodes-per-seed 1 --device cpu
python test.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

4. Optional policy walking video, about 1-2 minutes.

## Optional Walking Video

The assignment notice mainly asks for segmented training/result recording. A
walking video is useful as extra evidence but should not replace the policy file
or raw reward evaluation.

Generate a final trajectory:

```bash
python export_trajectory.py --run-dir runs/local_sac_cpu_5m_seed3407 --checkpoint-step 5000000 --seed 123 --episodes 1 --device cpu
```

Render latest trajectory:

```bash
python render_latest_trajectory.py --run-dir runs/local_sac_cpu_5m_seed3407 --backend egl --fps 20
```

If running locally on Windows and `egl` fails, use the already exported
trajectory on Colab, or simply submit the terminal/result recording and
screenshots.

The generated video should be under:

```text
runs/local_sac_cpu_5m_seed3407/trajectories/videos/
```

## Files To Protect

Do not delete:

```text
runs/local_sac_cpu_5m_seed3407/models/checkpoint_model_5000000_steps.zip
runs/local_sac_cpu_5m_seed3407/config.json
runs/local_sac_cpu_5m_seed3407/metadata.json
runs/local_sac_cpu_5m_seed3407/evaluations/raw_eval_20260608_101517.json
runs/local_sac_cpu_5m_seed3407/evaluations/raw_eval_20260608_101517.csv
```
