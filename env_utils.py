import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor    

def make_car_env(render_mode=None,  num_envs=4):
    def make_env():
        env = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            continuous=False,
        )
        env = Monitor(env)
        env = GrayscaleObservation(env, keep_dim=True)
        return env
        
    env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, 4)
    
    return env
def make_lunarlander_env(render_mode=None, num_envs=4):
    def make_env():
        env = gym.make(
            "LunarLander-v3", 
            render_mode=render_mode,
            continuous=False
        )
        env = Monitor(env)
        return env
    
    env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
    return env