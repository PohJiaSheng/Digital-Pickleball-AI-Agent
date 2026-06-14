from __future__ import annotations

import argparse
import multiprocessing as mp

import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder

from services.callbacks import UpdateOpponentCallback
from services.config import TrainConfig, load_config
from services.features import FEATURE_EXTRACTORS
from services.unity_wrappers import ActionFixedWrapper, SelfPlayUnityWrapper


def make_unity_env(config: TrainConfig, rank: int):
    def _init():
        string_channel = StringSideChannel()
        data_channel = CustomDataChannel()
        data_channel.send_data(serve=config.unity.serve_code, p1=0, p2=0)

        additional_args = [
            "-logFile",
            config.unity.log_file,
            "-batchmode",
            "-timeScale",
            str(config.unity.time_scale),
        ]

        unity_env = UnityEnvironment(
            file_name=config.paths.unity_executable,
            no_graphics=config.unity.no_graphics,
            side_channels=[string_channel, data_channel],
            additional_args=additional_args,
            worker_id=rank,
            base_port=config.unity.base_port + rank,
        )

        env = SelfPlayUnityWrapper(
            unity_env,
            frame_stack=config.env.frame_stack,
            img_size=config.env.image_size,
            grayscale=config.env.grayscale,
            action_control=config.env.action_control,
            device=config.env.device,
        )
        return ActionFixedWrapper(env, action_control=config.env.action_control)

    return _init


def build_model(config: TrainConfig, vec_env):
    learning_rate = get_schedule_fn(config.ppo.learning_rate)
    clip_range = get_schedule_fn(config.ppo.clip_range)

    if config.paths.load_path:
        model = PPO.load(
            config.paths.load_path,
            env=vec_env,
            device=torch.device(config.env.device),
            custom_objects={
                "learning_rate": learning_rate,
                "clip_range": clip_range,
                "n_steps": config.ppo.n_steps,
                "batch_size": config.ppo.batch_size,
                "n_epochs": config.ppo.n_epochs,
            },
            print_system_info=True,
        )
        print(f"Loaded checkpoint from {config.paths.load_path}")
        return model

    extractor_class = FEATURE_EXTRACTORS[config.ppo.extractor]
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {"features_dim": config.ppo.features_dim},
    }

    return PPO(
        "CnnPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gae_lambda=config.ppo.gae_lambda,
        gamma=config.ppo.gamma,
        clip_range=clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        verbose=3,
        device=torch.device(config.env.device),
    )


def train(config: TrainConfig):
    torch.set_float32_matmul_precision("high")

    if not config.paths.unity_executable:
        raise ValueError("Set paths.unity_executable in your config before training.")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.video_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_unity_env(config, i) for i in range(config.env.n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method=config.env.start_method)

    vec_env = VecVideoRecorder(
        vec_env,
        video_folder=str(config.video_dir),
        record_video_trigger=lambda step: step % config.logging.video_freq == 0,
        video_length=config.logging.video_length,
        name_prefix="ppo_sp",
    )

    try:
        model = build_model(config, vec_env)
        callbacks = CallbackList(
            [
                UpdateOpponentCallback(
                    update_freq=config.opponent_update_freq,
                    update_now=bool(config.paths.load_path),
                    device=config.env.device,
                ),
                CheckpointCallback(
                    save_freq=config.logging.checkpoint_save_freq,
                    save_path=str(config.checkpoint_dir),
                    name_prefix="ppo_sp",
                ),
            ]
        )

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not bool(config.paths.load_path),
        )

        final_path = config.experiment_dir / "ppo_sp_FINAL"
        model.save(str(final_path))
        print(f"Saved final model to {final_path}")
    finally:
        vec_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO self-play for Unity pickleball.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML training config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if config.env.start_method:
        mp.set_start_method(config.env.start_method, force=True)

    train(config)


if __name__ == "__main__":
    main()

