from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    run_root: Path = Path("runs")
    unity_executable: str = ""
    load_path: str | None = None


@dataclass
class UnityConfig:
    base_port: int = 5005
    serve_code: int = 13
    time_scale: float = 1.0
    no_graphics: bool = False
    log_file: str = "-"


@dataclass
class EnvConfig:
    n_envs: int = 32
    frame_stack: int = 4
    image_size: tuple[int, int] = (148, 64)
    grayscale: bool = False
    action_control: int = 3
    device: str = "cuda:0"
    start_method: str | None = None


@dataclass
class PPOConfig:
    learning_rate: float = 2.5e-4
    clip_range: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 256
    batch_size: int = 1024
    n_epochs: int = 4
    total_timesteps: int = 1_000_000_000
    multiply_timesteps_by_envs: bool = True
    features_dim: int = 512
    extractor: str = "impala"


@dataclass
class LoggingConfig:
    video_freq: int = 10_000
    video_length: int = 1_000
    checkpoint_save_freq: int = 25_000
    opponent_update_rollouts: int = 10


@dataclass
class TrainConfig:
    experiment_name: str = "ppo-pickleball-self-play"
    paths: PathsConfig = field(default_factory=PathsConfig)
    unity: UnityConfig = field(default_factory=UnityConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def experiment_dir(self) -> Path:
        return self.paths.run_root / self.experiment_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.experiment_dir / "checkpoints"

    @property
    def video_dir(self) -> Path:
        return self.experiment_dir / "videos"

    @property
    def total_timesteps(self) -> int:
        if self.ppo.multiply_timesteps_by_envs:
            return self.ppo.total_timesteps * self.env.n_envs
        return self.ppo.total_timesteps

    @property
    def opponent_update_freq(self) -> int:
        return self.env.n_envs * self.ppo.n_steps * self.logging.opponent_update_rollouts


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, Path):
            setattr(instance, key, Path(value))
        elif key == "image_size":
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path = "configs/default.yaml") -> TrainConfig:
    config = TrainConfig()
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return _merge_dataclass(config, data)

