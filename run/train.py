import argparse
import sys
import os
sys.path.append(os.getcwd())

from src.environment.game_env import GameEnv
from src.agent.reinforce_agent import ReinforceAgent
from src.training.trainer import Trainer
from src.training.logger import Logger
from src.utils.config import EnvConfig, AgentConfig, TrainConfig
from src.utils.seed import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--norm", action="store_true", help="Use return normalization")
    parser.add_argument("--entropy", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--state", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--reward", choices=["basic", "enhanced"], default="basic")
    parser.add_argument("--episodes", type=int, default=800)
    return parser.parse_args()

def main():
    args = parse_args()
    # Инициализация конфигов с учетом аргументов
    env_cfg = EnvConfig(state_mode=args.state, reward_mode=args.reward)
    agent_cfg = AgentConfig(use_normalization=args.norm, entropy_coef=args.entropy)
    
    # Настройка путей для эксперимента
    train_cfg = TrainConfig(
        num_episodes=args.episodes,
        exp_name=args.name,
        stats_path=f"artifacts/ablation/{args.name}/stats.csv",
        checkpoint_dir=f"artifacts/ablation/{args.name}/checkpoints"
    )

    print(f"\n>>> Running Experiment: {args.name}")
    print(f"Configs: Norm={args.norm}, Entropy={args.entropy}, State={args.state}, Reward={args.reward}")

    # воспроизводимость / установка seed 
    set_global_seed(train_cfg.seed)

    env = GameEnv(env_cfg)
    agent = ReinforceAgent(agent_cfg)
    logger = Logger(train_cfg.stats_path)
    trainer = Trainer(env, agent, train_cfg, logger)

    try:
        trainer.train()
    finally:
        logger.close()

if __name__ == "__main__":
    main()