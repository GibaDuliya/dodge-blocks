import argparse
import pygame
import numpy as np

# Правильные импорты согласно структуре
from src.environment.game_env import GameEnv
from src.environment.renderer import GameRenderer
from src.utils.config import EnvConfig, RenderConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "agent"], default="human")
    return parser.parse_args()

def play_human(env: GameEnv, renderer: GameRenderer) -> None:
    running = True
    
    while running:
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        
        done = False
        total_score = 0

        # Основной игровой цикл
        while not done:
            if not renderer.handle_events():
                return # Полный выход

            keys = pygame.key.get_pressed()
            action = 1 
            if keys[pygame.K_LEFT]: action = 0
            elif keys[pygame.K_RIGHT]: action = 2

            state, reward, done, info = env.step(action)
            total_score += reward
            renderer.render(state, int(total_score))

        # Цикл меню (когда игра окончена)
        waiting_for_input = True
        while waiting_for_input:
            renderer.render_menu(int(total_score)) # Рисуем меню
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Рестарт
                        waiting_for_input = False
                    if event.key == pygame.K_q: # Выход
                        running = False
                        waiting_for_input = False

    renderer.close()

def main():
    args = parse_args()
    
    # Инициализация конфигов
    e_cfg = EnvConfig()
    r_cfg = RenderConfig()
    
    # Инициализация среды и рендерера
    env = GameEnv(e_cfg)
    renderer = GameRenderer(e_cfg, r_cfg)

    if args.mode == "human":
        play_human(env, renderer)
    else:
        # Здесь будет логика для play_agent (Блок 3)
        pass

if __name__ == "__main__":
    main()