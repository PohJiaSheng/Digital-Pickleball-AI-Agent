# Unity Pickleball PPO Self-Play Agent

This project trains a PPO self-play agent for a Unity pickleball environment using Stable-Baselines3 and the Unity ML-Agents Python environment API. With this trained agent, we successfully secured a spot in the top three at the Digital Pickleball Competition.


## Environment Setup

1. Create and activate a virtual environment:

```bash
python -m venv pkball_env
source pkball_env/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

`Note`: Install the PyTorch build that matches your CUDA version if the default `torch` package does not match your GPU setup.

## Configuration

- `paths.unity_executable`: path to the Unity environment.
- `paths.run_root`: output folder for checkpoints and videos.
- `paths.load_path`: optional checkpoint path for resumed training.
- `env.n_envs`: number of parallel Unity workers.
- `env.device`: training device, for example `cuda:0` or `cpu`.
- `unity.serve_code`: environment-side serve or scenario code.
- `ppo.total_timesteps`: PPO training horizon before optional multiplication by `env.n_envs`.


## Training

```bash
python train.py 
```

Outputs:

```text
runs/<experiment_name>/
+-- checkpoints/
+-- videos/
`-- ppo_sp_FINAL.zip
```
