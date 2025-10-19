import time
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO, DQN
from env_utils import make_car_env


def watch_agent(model_path, method="ppo", episodes=3, fps=30, record=False, video_folder="videos/"):
    """
    Visualize a trained PPO or DQN agent playing the environment.

    Args:
        model_path (str): Path to the saved model (.zip)
        method (str): "ppo" or "dqn"
        env_name (str): "CarRacing-v3" or "LunarLander-v3"
        episodes (int): Number of episodes to play
        fps (int): Target frames per second for visualization
        record (bool): Whether to record gameplay to disk
        video_folder (str): Folder to save videos in (only if record=True)
    """
    print(f"Watching {method.upper()} agent for {episodes} episodes...")

    render_mode = "rgb_array" if record else "human"

    # Create environment
    env = make_car_env(render_mode=render_mode) if record else make_car_env(render_mode=render_mode, num_envs=1)

     # Wrap for video recording
    if record:
        env = VecVideoRecorder(
            env,
            video_folder=video_folder,
            record_video_trigger=lambda ep: True,
            video_length=1000000,
            name_prefix=f"{method}_car_racing",
        )

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device) if method.lower() == "ppo" else DQN.load(model_path, device=device)

    for ep in range(episodes):
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output

        done, truncated = False, False
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_output = env.step(action)

            if len(step_output) == 5:
                obs, reward, done, truncated, info = step_output
            else:
                obs, reward, done, info = step_output
                truncated = False

            total_reward += float(np.mean(reward))

            if not record:
                time.sleep(1 / fps)

            if np.any(done) or np.any(truncated):
                break

        print(f"Episode {ep + 1}/{episodes} — Total reward: {total_reward:.2f}")

    env.close()
    print("Visualization complete.")
    if record:
        print(f"Videos saved to: {video_folder}")
