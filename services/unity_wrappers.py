from __future__ import annotations

import functools
from collections import deque
from typing import Any

import cv2
import numpy as np
import torch
from gym import Env, Wrapper, spaces
from gym.utils import seeding
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv


class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack: int = 4, img_size=(148, 64), grayscale: bool = False):
        self.env = UnityParallelEnv(unity_env)

        self.agent = self.env.possible_agents[1]
        self.agent_other = self.env.possible_agents[0]
        self.agent_obs = self.env.possible_agents[0]
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        base_obs = self.env.observation_spaces[self.agent_obs][0]
        channels, _, _ = base_obs.shape
        self._transpose = channels == 3

        if grayscale:
            obs_shape = (frame_stack, img_size[1], img_size[0])
        else:
            obs_shape = (frame_stack * channels, img_size[1], img_size[0])

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32,
        )

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        obs = obs[:, 10:-10, 10:-10]
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        obs = (obs * 255).astype(np.uint8)

        threshold = 242
        agent_mask = np.where(obs[..., 0] > threshold, 1, 0).astype(np.uint8)

        border_mask = np.zeros_like(agent_mask)
        border_mask[4, 48:100] = 1
        border_mask[59, 48:100] = 1

        ball_mask_low = np.array([220, 223, 31], dtype=np.uint8)
        ball_mask_high = np.array([250, 250, 50], dtype=np.uint8)
        ball_mask = cv2.inRange(obs, ball_mask_low, ball_mask_high)
        ball_mask >>= 7

        masks = np.stack((agent_mask, border_mask, ball_mask), axis=0).astype(np.float32)
        if self.grayscale:
            return np.max(masks, axis=0, keepdims=True)
        return masks

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()
        obs = self._preprocess(obs_dict[self.agent_obs]["observation"][0])

        for _ in range(self.frame_stack):
            self.frames.append(obs)

        return np.concatenate(list(self.frames), axis=0), {}

    def step(self, action):
        actions = {self.agent: action}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        obs = self._preprocess(obs_dict[self.agent_obs]["observation"][0])
        self.frames.append(obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)
        reward = rewards[self.agent] - rewards[self.agent_other]
        done = terminations[self.agent]

        return stacked_obs, reward, done, False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class SelfPlayUnityWrapper(SharedObsUnityGymWrapper):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        unity_env,
        opponent_model: Any | None = None,
        frame_stack: int = 4,
        img_size=(148, 64),
        grayscale: bool = False,
        action_control: int = 3,
        device: str = "cuda:0",
    ):
        super().__init__(unity_env, frame_stack, img_size, grayscale)

        self.device = torch.device(device)
        self.action_control = action_control
        self.opponent_policy = (
            opponent_model.to(self.device).eval()
            if opponent_model is not None
            else None
        )
        self.render_mode = "rgb_array"
        self.action_space = self.env.action_spaces[self.agent]

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")

        img = self.frames[-1]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return (img * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def flip_decorator(func):
        lut = np.array([0, 2, 1], dtype=np.int32)

        @functools.wraps(func)
        def wrapper(self, obs):
            action = func(self, obs[..., ::-1])
            action[1:] = lut[action[1:]]
            return action

        return wrapper

    def _expand_to_xyz(self, action):
        if self.action_control == 1:
            return np.array([int(action), 0, 0], dtype=np.int32)
        if self.action_control == 2:
            return np.array([action[0], action[1], 0], dtype=np.int64)
        return action

    @flip_decorator
    def predict_opponent_action(self, stacked_obs):
        if self.opponent_policy is None:
            if self.action_control == 1:
                raw_action = np.random.randint(0, 3)
            elif self.action_control == 2:
                raw_action = np.random.randint(0, 3, size=2)
            else:
                raw_action = self.action_space.sample()
        else:
            with torch.no_grad():
                raw_action, _ = self.opponent_policy.predict(stacked_obs, deterministic=False)
                raw_action = raw_action.squeeze(0)

        return self._expand_to_xyz(raw_action)

    def step(self, action):
        opponent_action = self.predict_opponent_action(
            np.concatenate(list(self.frames), axis=0)[None, ...]
        )

        actions = {self.agent: action, self.agent_other: opponent_action}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        obs = self._preprocess(obs_dict[self.agent_obs]["observation"][0])
        self.frames.append(obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)
        reward = rewards[self.agent] - rewards[self.agent_other]
        done = terminations[self.agent]

        return stacked_obs, reward, done, False, infos[self.agent]

    def set_opponent_policy(self, policy):
        self.opponent_policy = policy.to(self.device).eval()


class ActionFixedWrapper(Wrapper):
    def __init__(self, env, action_control: int = 3):
        super().__init__(env)
        self.action_control = action_control

        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise TypeError("Expected a MultiDiscrete Unity action space.")

        if action_control == 1:
            self.action_space = spaces.Discrete(3)
        elif action_control == 2:
            self.action_space = spaces.MultiDiscrete([3, 3])
        elif action_control != 3:
            raise ValueError("action_control must be 1, 2, or 3.")

    def step(self, action):
        if self.action_control == 1:
            full_action = np.array([int(action), 0, 0], dtype=np.int32)
            return self.env.step(full_action)
        if self.action_control == 2:
            x_action, y_action = map(int, action)
            full_action = np.array([x_action, y_action, 0], dtype=np.int32)
            return self.env.step(full_action)
        return self.env.step(action)
