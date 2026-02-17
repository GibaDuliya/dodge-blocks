import argparse
import pygame
import torch
import sys
import os
import time

sys.path.append(os.getcwd())

from src.environment.game_env import GameEnv
from src.environment.renderer import GameRenderer
from src.agent.reinforce_agent import ReinforceAgent
from src.utils.config import EnvConfig, RenderConfig, AgentConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "agent"], default="human")
    parser.add_argument("--model", default="best.pt")
    return parser.parse_args()

def play_human(env, renderer):
    running = True
    while running:
        env.reset()
        done = False
        total_score = 0
        while not done:
            if not renderer.handle_events(): return
            keys = pygame.key.get_pressed()
            action = 1 
            if keys[pygame.K_LEFT]: action = 0
            elif keys[pygame.K_RIGHT]: action = 2

            _, reward, done, _ = env.step(action)
            total_score += reward
            renderer.render(env, int(total_score)) # Передаем env
        
        if not show_game_over(renderer, total_score): running = False
    renderer.close()

def play_agent(env, renderer, agent):
    running = True
    while running:
        state = env.reset()
        done = False
        total_score = 0
        print("New Episode Starting...")
        while not done:
            if not renderer.handle_events(): return

            with torch.no_grad():
                action = agent.select_action(state)
                agent.clear_buffers()

            state, reward, done, _ = env.step(action)
            total_score += reward
            renderer.render(env, int(total_score)) # Передаем env

        print(f"Agent Score: {total_score}")
        time.sleep(0.5) 
        if not show_game_over(renderer, total_score): running = False
    renderer.close()

def show_game_over(renderer, score):
    waiting = True
    while waiting:
        renderer.render_menu(int(score))
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: return True
                if event.key == pygame.K_q: return False
    return False

def main():
    args = parse_args()
    e_cfg = EnvConfig()
    r_cfg = RenderConfig()
    a_cfg = AgentConfig()
    
    seed = 42  # Default seed for play mode
    env = GameEnv(e_cfg, seed)
    renderer = GameRenderer(e_cfg, r_cfg)

    if args.mode == "human":
        play_human(env, renderer)
    else:
        agent = ReinforceAgent(a_cfg)
        if args.model.startswith("artifacts/"):
            ckpt_path = args.model
        else:
            ckpt_path = os.path.join("artifacts/checkpoints", args.model)
        
        if not os.path.exists(ckpt_path):
            print(f"Error: Model file not found at {ckpt_path}")
            return

        agent.load(ckpt_path)
        print(f"Successfully loaded model: {ckpt_path}")
        
        try:
            play_agent(env, renderer, agent)
        except Exception as e:
            print(f"Game crashed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()