import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor    

class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    Convert continuous CarRacing actions to discrete actions.
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

def make_car_env(render_mode=None, discretized=False, num_envs=4):
    def make_env():
        env = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            continuous=False,
        )
        env = Monitor(env)
        env = GrayscaleObservation(env, keep_dim=True)
        if discretized:
            env = DiscretizeActionWrapper(env)
        return env
        
    env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, 4)
    
    return env
def make_lunarlander_env(render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    env = Monitor(env)
    return env