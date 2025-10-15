import os
import json
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_env

def train_dqn(total_timesteps=300_000, log_dir="logs/dqn",
              model_path="models/dqn_car_racing.zip", optuna_params_path=None, n_envs=4):
    """
    Tränar en DQN-agent på diskretiserad CarRacing-v3.

    Args:
        total_timesteps (int): Antal träningssteg.
        log_dir (str): TensorBoard-loggning.
        model_path (str): Sparad modellväg.
        optuna_params_path (str): Valfri JSON-fil med Optuna-hyperparametrar.
    """
    print("Initializing DQN training environment...")
    env = make_car_env(num_envs=n_envs, discretized=True, render_mode=None, grayscale=True)

    # Ladda Optuna-hyperparametrar om fil finns
    if optuna_params_path and os.path.exists(optuna_params_path):
        with open(optuna_params_path, "r") as f:
            params = json.load(f)
        print("Using Optuna hyperparameters:", params)
    else:
        params = {}

    print("Setting up DQN model...")
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=params.get("learning_rate", 1e-4),
        gamma=params.get("gamma", 0.99),
        batch_size=params.get("batch_size", 32),
        buffer_size=params.get("buffer_size", 50000),
        tau=params.get("tau", 0.8),
        train_freq=params.get("train_freq", 4),
        target_update_interval=params.get("target_update_interval", 1000)
    )

    print(f"🏋️ Training DQN for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"DQN model saved at {model_path}")

    env.close()
    return model


def evaluate_dqn(model_path="models/dqn_car_racing.zip", episodes=5, render=True):
    """
    Utvärderar en sparad DQN-agent.

    Args:
        model_path (str): Sparad modell.
        episodes (int): Antal utvärderingsavsnitt.
        render (bool): Om True, rendera miljön.
    """
    print(f"Evaluating DQN model ({episodes} episodes)...")
    env = make_car_env(num_envs=1, discretized=True, render_mode="human" if render else None)
    model = DQN.load(model_path)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, render=render)
    print(f"DQN Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward
