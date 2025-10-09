import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium.spaces import Box

class CustomFrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        shape = env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=(shape[0], shape[1], shape[2] * num_stack), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=-1), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=-1), reward, terminated, truncated, info
