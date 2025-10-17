import time
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from env_utils import make_car_env, make_lunarlander_env


def watch_agent(model_path, method="ppo", env_name="CarRacing-v3", episodes=3, fps=30):
    """
    Visualize a trained PPO or DQN agent playing the environment.

    Args:
        model_path (str): Path to the saved model (.zip)
        method (str): "ppo" or "dqn"
        env_name (str): "CarRacing-v3" or "LunarLander-v3"
        episodes (int): Number of episodes to play
        fps (int): Target frames per second for visualization
    """
    print(f"🎬 Watching {method.upper()} agent on {env_name} for {episodes} episodes...")

    # Select the right environment (render_mode='human' for visualization)
    if env_name == "CarRacing-v3":
        env = make_car_env(render_mode="human", num_envs=1)
    elif env_name == "LunarLander-v3":
        env = make_lunarlander_env(render_mode="human")
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if method.lower() == "ppo":
        model = PPO.load(model_path, device=device)
    elif method.lower() == "dqn":
        model = DQN.load(model_path, device=device)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Run multiple episodes visually
    for ep in range(episodes):
        # Handle Gymnasium vs SB3 vectorized reset signatures
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            obs = reset_output[0]
        else:
            obs = reset_output

        done, truncated = False, False
        total_reward = 0.0

        while True:
            # Predict action deterministically
            action, _ = model.predict(obs, deterministic=True)

            # Step the environment (handle Gym vs Gymnasium step outputs)
            step_output = env.step(action)
            if len(step_output) == 5:
                obs, reward, done, truncated, info = step_output
            else:
                # Old-style SB3 VecEnv: (obs, reward, done, info)
                obs, reward, done, info = step_output
                truncated = False

            total_reward += float(np.mean(reward))

            # Handle vectorized "done" arrays
            if np.any(done) or np.any(truncated):
                break

            time.sleep(1 / fps)

        print(f"🎯 Episode {ep + 1}/{episodes} finished — Total reward: {total_reward:.2f}")

    env.close()
    print("✅ Visualization complete.")
