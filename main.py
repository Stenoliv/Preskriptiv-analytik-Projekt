import argparse
from train_ppo import train_ppo, evaluate_ppo
from train_dqn import train_dqn, evaluate_dqn
from visualize import watch_agent
from optuna_tune import run_optuna

def main():
    parser = argparse.ArgumentParser(description="CarRacing RL Project")

    parser.add_argument(
        "mode",
        choices=[
            "train-ppo", "evaluate-ppo", "watch-ppo",
            "train-dqn", "evaluate-dqn", "watch-dqn"
        ],
        help="What to run"
    )

    # Generella argument
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation/Watch episodes")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model")
    parser.add_argument("--optuna_best", type=str, default=None, help="Path to Optuna JSON file for hyperparameters")

    # Optuna-sökning
    parser.add_argument("--optuna", choices=["ppo", "dqn"], help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--optuna_timesteps", type=int, default=50_000, help="Timesteps per trial for Optuna")

    args = parser.parse_args()

    if args.optuna:
        run_optuna(method=args.optuna, n_trials=args.trials, timesteps=args.optuna_timesteps)
        return  # avsluta efter Optuna

    if not args.model_path:
        if args.mode.startswith("ppo"):
            args.model_path = "models/ppo_car_racing.zip"
        elif args.mode.startswith("dqn"):
            args.model_path = "models/dqn_car_racing.zip"

    if args.mode == "train-ppo":
        train_ppo(total_timesteps=args.timesteps, optuna_params_path=args.optuna_best)

    elif args.mode == "evaluate-ppo":
        evaluate_ppo(model_path=args.model_path, episodes=args.episodes, render=True)

    elif args.mode == "watch-ppo":
        watch_agent(model_path=args.model_path, method="ppo", episodes=args.episodes)

    elif args.mode == "train-dqn":
        train_dqn(total_timesteps=args.timesteps, optuna_params_path=args.optuna_best)

    elif args.mode == "evaluate-dqn":
        evaluate_dqn(model_path=args.model_path, episodes=args.episodes, render=True)

    elif args.mode == "watch-dqn":
        watch_agent(model_path=args.model_path, method="dqn", episodes=args.episodes)

if __name__ == "__main__":
    main()
