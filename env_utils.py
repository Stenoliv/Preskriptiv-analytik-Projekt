# env_utils.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

class DiscretizedCarRacing(gym.Wrapper):
    """
    Wrapper som gör CarRacing till ett diskret actions-space för DQN.
    """
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            np.array([0.0, 0.0, 0.0]),   # inget
            np.array([0.0, 1.0, 0.0]),   # gas
            np.array([0.0, 0.0, 0.8]),   # broms
            np.array([-1.0, 1.0, 0.0]),  # vänster + gas
            np.array([1.0, 1.0, 0.0]),   # höger + gas
        ]
        self.action_space = spaces.Discrete(len(self.actions))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.actions[action])
        return obs, reward, terminated, truncated, info


def make_car_racing_env(discretized=False, grayscale=True, resize_shape=(64, 64), render_mode=None):
    """
    Skapar och returnerar en CarRacing-v2 miljö.
    """
    env = gym.make("CarRacing-v2", render_mode=render_mode)

    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    if resize_shape:
        env = ResizeObservation(env, resize_shape)
    if discretized:
        env = DiscretizedCarRacing(env)

    return env
