import argparse
import sys
import os
import torch
sys.path.append(os.getcwd())

from src.environment.game_env import GameEnv
from src.agent.reinforce_agent import ReinforceAgent
from src.utils.config import EnvConfig, AgentConfig
from src.utils.seed import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    set_global_seed(args.seed)
    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()
    
    # Create environment and agent
    env = GameEnv(env_cfg, args.seed)
    agent = ReinforceAgent(agent_cfg)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluation
    total_rewards = []
    total_steps = []
    
    print(f"\nEvaluating agent for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        
        while not done:
            with torch.no_grad():
                action = agent.select_action(state)
                agent.clear_buffers()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(total_rewards[-10:]) / 10
            avg_steps = sum(total_steps[-10:]) / 10
            print(f"Episode {episode + 1}/{args.num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.1f}")
    
    # Final statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_steps = sum(total_steps) / len(total_steps)
    max_reward = max(total_rewards)
    min_reward = min(total_rewards)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Max Reward: {max_reward:.2f}")
    print(f"  Min Reward: {min_reward:.2f}")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
