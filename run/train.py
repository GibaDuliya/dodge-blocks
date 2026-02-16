import sys
import os

# Добавляем корневую папку в путь, чтобы видеть модули src
sys.path.append(os.getcwd())

from src.environment.game_env import GameEnv
from src.agent.reinforce_agent import ReinforceAgent
from src.training.trainer import Trainer
from src.training.logger import Logger
# Импортируем классы конфигов (значения уже внутри них)
from src.utils.config import EnvConfig, AgentConfig, TrainConfig

def main():
    # 1. Инициализация конфигов
    # Значения берутся из defaults в src/utils/config.py
    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()
    train_cfg = TrainConfig()

    print(f"Loaded Configuration:")
    print(f"- Grid: {env_cfg.grid_width}x{env_cfg.grid_height}")
    print(f"- Episodes: {train_cfg.num_episodes}")
    print(f"- Learning Rate: {agent_cfg.learning_rate}")

    # 2. Создание объектов
    env = GameEnv(env_cfg)
    agent = ReinforceAgent(agent_cfg)
    logger = Logger(train_cfg.stats_path)
    
    trainer = Trainer(env, agent, train_cfg, logger)

    # 3. Запуск
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")
    finally:
        logger.close()
        print("Logs saved.")

if __name__ == "__main__":
    main()