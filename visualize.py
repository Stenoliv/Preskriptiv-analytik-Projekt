from stable_baselines3 import PPO, DQN
from env_utils import make_car_racing_env

def watch_agent(model_path, method="ppo", episodes=3):
    """
    Kör agenten visuellt efter träning.
    
    Args:
        model_path (str): Sökväg till sparad modell
        method (str): 'ppo' eller 'dqn'
        episodes (int): Antal körningar
    """
    render_mode = "human"  # visa miljön
    discretized = method.lower() == "dqn"
    env = make_car_racing_env(discretized=discretized, render_mode=render_mode)

    if method.lower() == "ppo":
        model = PPO.load(model_path)
    elif method.lower() == "dqn":
        model = DQN.load(model_path)
    else:
        raise ValueError("Method must be 'ppo' or 'dqn'")

    for ep in range(episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {ep+1}: total reward = {total_reward:.2f}")

    env.close()
