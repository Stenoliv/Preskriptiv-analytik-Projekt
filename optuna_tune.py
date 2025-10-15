import os
import optuna
import json
import torch
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from env_utils import make_car_env

def optuna_objective_ppo(trial, timesteps=50_000, envs=4):
    """Optuna objective for PPO"""
    import time
    from stable_baselines3.common.monitor import Monitor
    import numpy as np
    
    mean_reward = -9999.0  # ✅ default value
    start_time = time.time()
    trial_number = trial.number
    print(f"\n🧠 Starting PPO trial #{trial_number}")
    print("-" * 60)
    
    # --- Hyperparameter search space ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    print(f"🧩 Params: lr={learning_rate:.1e}, gamma={gamma}, n_steps={n_steps}, batch_size={batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env = make_car_env(render_mode=None, num_envs=envs, discretized=False, grayscale=True, resize_shape=(48, 48))
    eval_env = make_car_env(render_mode=None, num_envs=1, discretized=False, grayscale=True, resize_shape=(48, 48))
    
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        clip_range=clip_range,
        device=device
    )
    
    # Split the training into chunks so we can report progress
    steps_done = 0
    report_every = max(10_000, timesteps // 10)

    try:
        while steps_done < timesteps:
            model.learn(total_timesteps=report_every, reset_num_timesteps=False)
            steps_done += report_every
            
            # Evaluate current model
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=2)
            if isinstance(mean_reward, (list, tuple)):
                mean_reward = float(np.mean(mean_reward))
            
            print(f"🪶 Trial #{trial_number} @ {steps_done:,} steps — mean_reward={mean_reward:.2f}")
            
            # Report intermediate objective value
            trial.report(mean_reward, steps_done)
            
            # Check if this trial should be pruned
            if trial.should_prune():
                print(f"⛔ Pruned trial #{trial_number} at {steps_done:,} steps")
                raise optuna.TrialPruned()
    except Exception as e:
        print(f"[Trial failed] {e}")
        mean_reward = -9999
    finally:
        env.close()
        eval_env.close()
        
    print(f"✅ Trial #{trial_number} completed in {time.time() - start_time:.1f}s with reward {mean_reward:.2f}")
    print("=" * 60)
    
    return mean_reward

def optuna_objective_dqn(trial, timesteps=50_000, envs=1):
    """Optuna objective for DQN"""
    env = make_car_env(render_mode=None, num_envs=envs, discretized=False, grayscale=True)

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

    # 🧠 Export-only mode
    if method.lower() == "export-only":
        print("📤 Exporting best parameters from existing PPO study...")
        study = optuna.load_study(study_name="ppo_car_racing", storage="sqlite:///optuna_ppo.db")
        best_params = study.best_params

        os.makedirs(os.path.dirname(save_json), exist_ok=True)

        with open(save_json, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"💾 Exported best parameters to {save_json}")
        return best_params

    # 🧩 Normal optimization mode
    print(f"🚀 Running Optuna for {method.upper()} with {n_trials} trials × {timesteps:,} timesteps")
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name, 
        storage=storage, 
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )

    if method.lower() == "ppo":
        study.optimize(lambda trial: optuna_objective_ppo(trial, timesteps, envs), n_jobs=max(1, int(envs / 4)), n_trials=n_trials)
    elif method.lower() == "dqn":
        study.optimize(lambda trial: optuna_objective_dqn(trial, timesteps, envs), n_jobs=max(1, int(envs / 4)), n_trials=n_trials)
    else:
        raise ValueError("Method must be 'ppo', 'dqn', or 'export-only'")

    best_params = study.best_params
    print(f"\n✅ Best hyperparameters for {method.upper()}:")
    print(json.dumps(best_params, indent=2))

    os.makedirs(os.path.dirname(save_json), exist_ok=True)

    with open(save_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"💾 Saved best parameters to {save_json}")

    return best_params
