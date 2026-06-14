from __future__ import annotations

import copy

import torch
from stable_baselines3.common.callbacks import BaseCallback


class UpdateOpponentCallback(BaseCallback):
    def __init__(
        self,
        update_freq: int = 50_000,
        verbose: int = 1,
        device: str = "cuda:0",
        update_now: bool = False,
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.device = torch.device(device)
        self.update_now = update_now

    def _on_step(self) -> bool:
        if not self.update_now and self.num_timesteps % self.update_freq != 0:
            return True

        self.update_now = False
        opponent_policy = copy.deepcopy(self.model.policy).to(self.device)
        opponent_policy.eval()
        self.training_env.env_method("set_opponent_policy", opponent_policy)

        if self.verbose:
            print(f"\n[Self-Play] opponent synced @ {self.num_timesteps:,} steps")
        return True

