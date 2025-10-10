# Preskriptiv-analytik-Projekt

## CarRacing RL Project - Kommandon

| Funktion                  | Kommando                                                                       | Kommentar                                                                  |
| ------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| **Träna PPO-agent**       | `python main.py train-ppo --timesteps 200000`                                  | Sparar modell till `models/ppo_car_racing.zip` och loggar till TensorBoard |
| **Evaluera PPO-agent**    | `python main.py evaluate-ppo --episodes 5`                                     | Renderar spelet och visar genomsnittlig reward                             |
| **Visualisera PPO-agent** | `python main.py watch-ppo --model_path models/ppo_car_racing.zip --episodes 3` | Kör agenten i render-läge för visuell feedback                             |
| **Träna DQN-agent**       | `python main.py train-dqn --timesteps 300000`                                  | Diskretiserad miljö, sparar modell till `models/dqn_car_racing.zip`        |
| **Evaluera DQN-agent**    | `python main.py evaluate-dqn --episodes 5`                                     | Renderar spelet och visar genomsnittlig reward                             |
| **Visualisera DQN-agent** | `python main.py watch-dqn --model_path models/dqn_car_racing.zip --episodes 3` | Kör agenten i render-läge för visuell feedback                             |
| **Optuna hyperparam PPO** | `python main.py --optuna ppo --trials 10 --optuna_timesteps 50000`             | Optimerar PPO, sparar bästa parametrar i `models/optuna_best.json`         |
| **Optuna hyperparam DQN** | `python main.py --optuna dqn --trials 10 --optuna_timesteps 50000`             | Optimerar DQN, sparar bästa parametrar i `models/optuna_best.json`         |
| **Starta TensorBoard**    | `tensorboard --logdir logs/`                                                   | Öppna `http://localhost:6006/` i webbläsare                                |
