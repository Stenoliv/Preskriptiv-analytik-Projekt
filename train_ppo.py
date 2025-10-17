import os
import json
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_env, make_lunarlander_env


def train_ppo(total_timesteps=200_000, n_envs=4, log_dir="logs/ppo",
              model_path="models/ppo_car_racing.zip", optuna_params_path=None,
              env_name="CarRacing-v3"):
    """
    Train a PPO agent on CarRacing-v3 or LunarLander-v3.
    """
    print(f"🚀 Initializing PPO training environment for {env_name}...")

    # Select environment
    if env_name == "CarRacing-v3":
        env = make_car_env(render_mode="rgb_array", num_envs=n_envs)
        policy = "CnnPolicy"
    else:
        env = DummyVecEnv([lambda: make_lunarlander_env() for _ in range(n_envs)])
        policy = "MlpPolicy"

    # Load Optuna hyperparameters if available
    if optuna_params_path and os.path.exists(optuna_params_path):
        with open(optuna_params_path, "r") as f:
            params = json.load(f)
        print("📊 Using Optuna hyperparameters:", params)
    else:
        params = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🧠 Setting up PPO model...")
    model = PPO(
        policy,
        env,
        verbose=1,
        n_steps=params.get("n_steps", 1024),
        batch_size=params.get("batch_size", 64),
        n_epochs=(10),
        gamma=params.get("gamma", 0.99),
        gae_lambda=(0.95),
        device=device,
        tensorboard_log=log_dir,
    )

    print(f"🏋️ Training PPO on {env_name} for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"✅ PPO model saved at {model_path}")

    env.close()
    return model


def evaluate_ppo(model_path="models/ppo_car_racing.zip", episodes=5, render=True, env_name="CarRacing-v3"):
    """
    Evaluate a trained PPO model.
    """
    print(f"🎮 Evaluating PPO model on {env_name} ({episodes} episodes)...")

    if env_name == "CarRacing-v3":
        env = make_car_env(render_mode="human" if render else None)
        policy = "CnnPolicy"
    else:
        env = make_lunarlander_env(render_mode="human" if render else None)
        policy = "MlpPolicy"

    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, render=render)
    print(f"📈 PPO Mean reward on {env_name}: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward
