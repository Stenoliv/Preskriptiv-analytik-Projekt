import optuna
import json
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_racing_env

def make_eval_env():
    """Return a monitored evaluation environment."""
    env = make_car_racing_env(render_mode=None)
    return Monitor(env)

def optuna_objective_ppo(trial, timesteps=50_000, n_envs=1):
    """Optuna objective for PPO"""
    env = SubprocVecEnv([lambda: make_eval_env() for _ in range(n_envs)])
    env = VecTransposeImage(env)

    # --- Hyperparameter search space ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    n_steps = trial.suggest_int("n_steps", 64, 2048, log=True)
    n_steps = max(64, int(n_steps // n_envs) * n_envs) # balance across envs
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        clip_range=clip_range,
        device=device
    )

    eval_env = SubprocVecEnv([lambda: make_eval_env() for _ in range(n_envs)])
    try:
        model.learn(total_timesteps=timesteps)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=3)
    except Exception as e:
        print(f"[Trial failed] {e}")
        mean_reward = -9999  # penalize failed runs
    finally:
        env.close()
        eval_env.close()
        
    return mean_reward


def optuna_objective_dqn(trial, timesteps=50_000, n_envs=1):
    """Optuna objective for DQN"""
    env = SubprocVecEnv([lambda: make_car_racing_env(discretized=True, render_mode=None) for _ in range(n_envs)])

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    buffer_size = trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000])
    tau = trial.suggest_float("tau", 0.5, 0.99)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
    target_update_interval = trial.suggest_categorical("target_update_interval", [500, 1000, 5000])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        target_update_interval=target_update_interval,
        device=device,
    )

    try:
        model.learn(total_timesteps=timesteps)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)
    except Exception as e:
        print(f"[Trial failed] {e}")
        mean_reward = -9999
    finally:
        env.close()
    
    return mean_reward

def run_optuna(method="ppo", n_trials=10, timesteps=50_000, envs=1, save_json="models/optuna_best.json"):
    """Run Optuna hyperparameter search for PPO or DQN."""
    study_name = f"{method}_car_racing"
    storage = f"sqlite:///optuna_{method}.db"  # persistent study

    print(f"Running Optuna for {method.upper()} with {n_trials} trials × {timesteps:,} timesteps")

    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True)

    if method.lower() == "ppo":
        study.optimize(lambda trial: optuna_objective_ppo(trial, timesteps, n_envs=envs), n_trials=n_trials)
    elif method.lower() == "dqn":
        study.optimize(lambda trial: optuna_objective_dqn(trial, timesteps, n_envs=envs), n_trials=n_trials)
    else:
        raise ValueError("Method must be 'ppo' or 'dqn'")

    best_params = study.best_params
    print(f"\n✅ Best hyperparameters for {method.upper()}:")
    print(json.dumps(best_params, indent=2))

    with open(save_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"💾 Saved best parameters to {save_json}")

    return best_params
