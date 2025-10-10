# optuna_tune.py
import optuna
import json
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_racing_env

def optuna_objective_ppo(trial, timesteps=50_000):
    """Optuna objective for PPO"""
    env = DummyVecEnv([lambda: make_car_racing_env(render_mode=None)])
    env = VecTransposeImage(env)

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    n_steps = int(trial.suggest_loguniform("n_steps", 64, 2048))
    batch_size = int(trial.suggest_categorical("batch_size", [32, 64, 128]))
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        clip_range=clip_range
    )

    model.learn(total_timesteps=timesteps)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)
    env.close()
    return mean_reward


def optuna_objective_dqn(trial, timesteps=50_000):
    """Optuna objective for DQN"""
    env = DummyVecEnv([lambda: make_car_racing_env(discretized=True, render_mode=None)])

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    batch_size = int(trial.suggest_categorical("batch_size", [32, 64, 128]))
    buffer_size = int(trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]))
    tau = trial.suggest_float("tau", 0.5, 0.99)
    train_freq = int(trial.suggest_categorical("train_freq", [1, 4, 8]))
    target_update_interval = int(trial.suggest_categorical("target_update_interval", [500, 1000, 5000]))

    model = DQN(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        tau=tau,
        train_freq=train_freq,
        target_update_interval=target_update_interval
    )

    model.learn(total_timesteps=timesteps)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)
    env.close()
    return mean_reward


def run_optuna(method="ppo", n_trials=10, timesteps=50_000, save_json="models/optuna_best.json"):
    """Run Optuna hyperparameter search for PPO or DQN"""
    if method.lower() == "ppo":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optuna_objective_ppo(trial, timesteps), n_trials=n_trials)
    elif method.lower() == "dqn":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optuna_objective_dqn(trial, timesteps), n_trials=n_trials)
    else:
        raise ValueError("Method must be 'ppo' or 'dqn'")

    best_params = study.best_params
    with open(save_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"✅ Best hyperparameters for {method} saved to {save_json}")
    return best_params
