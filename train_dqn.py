import os
import json
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_env, make_lunarlander_env


def train_dqn(total_timesteps=300_000, log_dir="logs/dqn",
              model_path="models/dqn_lunar_lander.zip", optuna_params_path=None,
              env_name="LunarLander-v3"):
    """
    Train a DQN agent on LunarLander-v3 or CarRacing-v3 (discrete version).
    """
    print(f"🚀 Initializing DQN training environment for {env_name}...")

    # DQN only works with discrete actions
    if env_name == "LunarLander-v3":
        env = DummyVecEnv([make_lunarlander_env])
    else:
        raise ValueError("❌ DQN does not support continuous action spaces like CarRacing-v2.")

    # Load Optuna parameters if available
    if optuna_params_path and os.path.exists(optuna_params_path):
        with open(optuna_params_path, "r") as f:
            params = json.load(f)
        print("📊 Using Optuna hyperparameters:", params)
    else:
        params = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🧠 Setting up DQN model...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=params.get("learning_rate", 1e-4),
        buffer_size=params.get("buffer_size", 100_000),
        learning_starts=params.get("learning_starts", 1000),
        batch_size=params.get("batch_size", 64),
        tau=params.get("tau", 1.0),
        gamma=params.get("gamma", 0.99),
        train_freq=params.get("train_freq", 4),
        target_update_interval=params.get("target_update_interval", 1000),
        device=device
    )

    print(f"🏋️ Training DQN on {env_name} for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"✅ DQN model saved at {model_path}")

    env.close()
    return model


def evaluate_dqn(model_path="models/dqn_lunar_lander.zip", episodes=5, render=True, env_name="LunarLander-v3"):
    """
    Evaluate a trained DQN model.
    """
    print(f"🎮 Evaluating DQN model on {env_name} ({episodes} episodes)...")

    if env_name != "LunarLander-v2":
        raise ValueError("❌ DQN evaluation only supported for LunarLander-v3 (discrete).")

    env = make_lunarlander_env(render_mode="human" if render else None)
    model = DQN.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, render=render)
    print(f"📈 DQN Mean reward on {env_name}: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward
