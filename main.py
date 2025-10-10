# main.py
import argparse
from train_ppo import train_ppo, evaluate_ppo
from train_dqn import train_dqn, evaluate_dqn
from optuna_tune import run_optuna
from visualize import watch_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CarRacing RL Project")
    parser.add_argument("mode", choices=["train-ppo", "train-dqn", "evaluate-ppo", "evaluate-dqn","watch-ppo", "watch-dqn"], help="What to run")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes")

    # Optuna hyperparameter tuning arguments
    parser.add_argument("--optuna", choices=["ppo", "dqn"], help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--optuna_timesteps", type=int, default=50_000, help="Timesteps per trial for Optuna")


    args = parser.parse_args()
    # Run Optuna if requested
    if args.optuna:
        run_optuna(method=args.optuna, n_trials=args.trials, timesteps=args.optuna_timesteps)
        exit(0)  # stoppar programmet efter Optuna-sökning


    if args.mode == "train-ppo":
        train_ppo(total_timesteps=args.timesteps)
    elif args.mode == "evaluate-ppo":
        evaluate_ppo(episodes=args.episodes)
    elif args.mode == "train-dqn":
        train_dqn(total_timesteps=args.timesteps)
    elif args.mode == "evaluate-dqn":
        evaluate_dqn(episodes=args.episodes)
    elif args.mode == "watch-ppo":
        watch_agent(args.model_path, method="ppo", episodes=args.episodes)
    elif args.mode == "watch-dqn":
        watch_agent(args.model_path, method="dqn", episodes=args.episodes)

