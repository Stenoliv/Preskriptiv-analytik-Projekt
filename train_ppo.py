# train_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_racing_env
import os

def train_ppo(total_timesteps=200_000, n_envs=4, log_dir="logs/ppo", model_path="models/ppo_car_racing.zip"):
    print("🚗 Initializing PPO training environment...")
    env = DummyVecEnv([lambda: make_car_racing_env(render_mode=None) for _ in range(n_envs)])
    env = VecTransposeImage(env)

    print("🧠 Setting up PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    print(f"🏋️ Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"✅ PPO model saved at {model_path}")

    env.close()
    return model


def evaluate_ppo(model_path="models/ppo_car_racing.zip", episodes=5, render=True):
    print(f"🎮 Evaluating PPO model ({episodes} episodes)...")
    env = make_car_racing_env(render_mode="human" if render else None)
    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, render=render)
    print(f"✅ PPO Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward
