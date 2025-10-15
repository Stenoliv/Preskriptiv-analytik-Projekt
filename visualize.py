import torch
import numpy as np
from typing import cast
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from env_utils import make_car_env

def watch_agent(model_path, method="ppo", episodes=3):
    """
    Kör agenten visuellt efter träning.
    
    Args:
        model_path (str): Sökväg till sparad modell
        method (str): 'ppo' eller 'dqn'
        episodes (int): Antal körningar
    """
    render_mode = "human"  # visa miljön
    discretized = method.lower() == "dqn"

    # Create environment with same wrappers as training
    env = make_car_env(num_envs=1, render_mode=render_mode, grayscale=False, discretized=discretized, resize_shape=(64, 64))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if method.lower() == "ppo":
        model = PPO.load(model_path, env=env, device=device)
    elif method.lower() == "dqn":
        model = DQN.load(model_path, env=env, device=device)
    else:
        raise ValueError("Method must be 'ppo' or 'dqn'")

    total_reward = 0.0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)  # VecEnv: 4 outputs
            done = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)
            ep_reward += float(rewards[0]) if isinstance(rewards, (list, np.ndarray)) else float(rewards)

        print(f"Episode {ep+1} total reward: {ep_reward:.2f}")
        total_reward += ep_reward

    env.close()
