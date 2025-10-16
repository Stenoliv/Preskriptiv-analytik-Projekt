import os
import argparse
from train_ppo import train_ppo, evaluate_ppo
from train_dqn import train_dqn, evaluate_dqn
from visualize import watch_agent
from optuna_tune import run_optuna

def parse_resize_shape(s):
    # Convert string "(94,94)" or "94,94" into a tuple of ints
    s = s.replace("(", "").replace(")", "")
    return tuple(map(int, s.split(",")))


def main():
    parser = argparse.ArgumentParser(description="RL Project - PPO & DQN on CarRacing or LunarLander")

    parser.add_argument(
        "mode",
        choices=[
            "optuna-ppo", "optuna-dqn",
            "train-ppo", "evaluate-ppo", "watch-ppo",
            "train-dqn", "evaluate-dqn", "watch-dqn"
        ],
        help="What to run"
    )

    # === General arguments ===
    parser.add_argument("--env", type=str, default="CarRacing-v3",
                        choices=["CarRacing-v3", "LunarLander-v3"],
                        help="Select environment")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments (for PPO)")
    parser.add_argument("--resize_shape", type=parse_resize_shape, default=(64, 64))
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation/Watch episodes")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model")
    parser.add_argument("--optuna_best", type=str, default=None, help="Path to Optuna JSON file for hyperparameters")

    # === Optuna arguments ===
    parser.add_argument("--optuna", choices=["ppo", "dqn", "export-only"], help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--optuna_timesteps", type=int, default=50_000, help="Timesteps per trial for Optuna")

    args = parser.parse_args()

    # === Run Optuna if requested ===
    if args.optuna:
        run_optuna(method=args.optuna, n_trials=args.trials, envs=args.envs, timesteps=args.optuna_timesteps)
        return

    # === Set default model path if not provided ===
    if not args.model_path:
        os.makedirs("models", exist_ok=True)
        algo = "ppo" if "ppo" in args.mode else "dqn"
        env_short = args.env.lower().replace("-", "_")
        args.model_path = f"models/{algo}_{env_short}.zip"

    # === Ensure model exists for evaluation/watch modes ===
    if args.mode.startswith(("evaluate", "watch")) and not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file not found at {args.model_path}. Please train the model first."
        )

    # === MODE HANDLER ===
    if args.mode == "train-ppo":
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            optuna_params_path=args.optuna_best,
            model_path=args.model_path,
            resize_shape=args.resize_shape,
            env_name=args.env,
            log_dir=f"logs/ppo/{args.env.lower().replace('-', '_')}"
        )

    elif args.mode == "evaluate-ppo":
        evaluate_ppo(
            model_path=args.model_path,
            episodes=args.episodes,
            render=True,
            env_name=args.env
        )

    elif args.mode == "watch-ppo":
        watch_agent(model_path=args.model_path, method="ppo", episodes=args.episodes, env_name=args.env)

    elif args.mode == "train-dqn":
        train_dqn(
            total_timesteps=args.timesteps,
            optuna_params_path=args.optuna_best,
            model_path=args.model_path,
            env_name=args.env,
            log_dir=f"logs/dqn/{args.env.lower().replace('-', '_')}"
        )

    elif args.mode == "evaluate-dqn":
        evaluate_dqn(model_path=args.model_path, episodes=args.episodes, render=True, env_name=args.env)

    elif args.mode == "watch-dqn":
        watch_agent(model_path=args.model_path, method="dqn", episodes=args.episodes, env_name=args.env)


if __name__ == "__main__":
    main()
