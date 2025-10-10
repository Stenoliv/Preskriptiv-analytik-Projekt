# train_dqn.py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_racing_env
import os

def train_dqn(total_timesteps=300_000, log_dir="logs/dqn", model_path="models/dqn_car_racing.zip"):
    print("🚘 Initializing DQN training environment...")
    env = DummyVecEnv([lambda: make_car_racing_env(discretized=True, render_mode=None)])

    print("🧠 Setting up DQN model...")
    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=50_000,
        learning_starts=10_000,
        batch_size=32,
        tau=0.8,
        gamma=0.99,
        learning_rate=1e-4,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
    )

    print(f"🏋️ Training DQN for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"✅ DQN model saved at {model_path}")

    env.close()
    return model


def evaluate_dqn(model_path="models/dqn_car_racing.zip", episodes=5, render=True):
    print(f"🎮 Evaluating DQN model ({episodes} episodes)...")
    env = make_car_racing_env(discretized=True, render_mode="human" if render else None)
    model = DQN.load(model_path)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, render=render)
    print(f"✅ DQN Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward
