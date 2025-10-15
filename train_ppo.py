import os
import json
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from env_utils import make_car_env

def train_ppo(total_timesteps=200_000, n_envs=4, log_dir="logs/ppo",
              model_path="models/ppo_car_racing.zip", optuna_params_path=None, resize_shape=(64,64)):
    """
    Tränar en PPO-agent på CarRacing-v2.

    Args:
        total_timesteps (int): Antal träningssteg.
        n_envs (int): Antal miljöer i vektoriserad träning.
        log_dir (str): TensorBoard-loggning.
        model_path (str): Sparad modellväg.
        optuna_params_path (str): Valfri JSON-fil med Optuna-hyperparametrar.
    """
    print("Initializing PPO training environment...")
    # Training env
    env = make_car_env(num_envs=n_envs, discretized=False, resize_shape=resize_shape, grayscale=False)

    if optuna_params_path and os.path.exists(optuna_params_path):
        with open(optuna_params_path, "r") as f:
            params = json.load(f)
        print("Using Optuna hyperparameters:", params)
    else:
        params = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Setting up PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        gamma=params.get("gamma", 0.99),
        gae_lambda=params.get("gae_lambda", 0.95),
        clip_range=params.get("clip_range", 0.2),
        n_steps=params.get("n_steps", 2048),
        batch_size=params.get("batch_size", 64),
        learning_rate=params.get("learning_rate", 3e-4),
        ent_coef=params.get("ent_coef", 0.01),
        tensorboard_log=log_dir,
    )

    print(f"Training PPO with {n_envs} parallel environments for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"PPO model saved at {model_path}")

    env.close()
    return model

def evaluate_ppo(model_path="models/ppo_car_racing.zip", episodes=5, render=True):
    """
    Utvärderar en sparad PPO-agent.

    Args:
        model_path (str): Sparad modell.
        episodes (int): Antal utvärderingsavsnitt.
        render (bool): Om True, rendera miljön.
    """
    print(f"Evaluating PPO model ({episodes} episodes)...")
    env = make_car_env(num_envs=1, render_mode="human" if render else None)

    device = "cude" if torch.cuda.is_available() else "cpu"
    
    model = PPO.load(model_path, env=env, device=device)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, render=render)
    print(f"PPO Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward
