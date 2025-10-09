import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from framestackwrapper import CustomFrameStack

def make_env(render_mode=None):
    """
    Create a wrapped CarRacing environment.

    Args:
        render_mode: "human" for visualization, None for training.
    Returns:
        Wrapped Gymnasium environment.
    """
    env = gym.make("CarRacing-v3", render_mode=render_mode)

    env = ResizeObservation(env, (64, 64))          
    env = GrayscaleObservation(env, keep_dim=True)
    env = CustomFrameStack(env, num_stack=4)                      

    return env

def train_agent(total_timesteps=200_000, n_envs=4, log_dir="./logs_car_racing/"):
    """
    Train a PPO agent on CarRacing-v3.
    """
    print("Initializing training environment...")
    env = make_vec_env(make_env, n_envs=n_envs)

    print("Setting up PPO model...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    print(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training finished!")

    model.save("ppo_car_racing")
    print("Model saved as ppo_car_racing.zip")

    env.close()
    return model

def evaluate_agent(model_path="ppo_car_racing", episodes=1):
    """
    Run a trained PPO agent in human-rendered mode.
    """
    print(f"Evaluating model from: {model_path}.zip")
    env = make_env(render_mode="human")

    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep+1} reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    model = train_agent(
        total_timesteps=100_000,  
        n_envs=4,
    )

    evaluate_agent("ppo_car_racing", episodes=2)
