import torch
import os
import numpy as np
from src.environment.game_env import GameEnv
from src.agent.reinforce_agent import ReinforceAgent
from src.training.logger import Logger

class Trainer:
    def __init__(self, env, agent, train_config, logger) -> None:
        self.env = env
        self.agent = agent
        self.cfg = train_config
        self.logger = logger
        # Начинаем с очень низкого значения
        self.best_reward = -float('inf')
        
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    def train(self) -> dict:
        print(f"Starting training for {self.cfg.num_episodes} episodes...")
        
        running_reward = 0.0
        
        for episode in range(1, self.cfg.num_episodes + 1):
            reward, steps = self.run_episode()
            
            # Обновляем сеть
            loss = self.agent.update_policy()
            
            # Более быстрое обновление среднего (0.9 вместо 0.95), чтобы видеть прогресс
            if episode == 1:
                running_reward = reward
            else:
                running_reward = 0.1 * reward + 0.9 * running_reward

            # Логирование
            if episode % self.cfg.log_every == 0:
                self.logger.log_episode(episode, running_reward, steps, loss)
                
            # Сохраняем "Лучшую" модель
            # Используем running_reward, чтобы отсеять случайные удачи
            if running_reward > self.best_reward and episode > 100:
                self.best_reward = running_reward
                self.save_model("best.pt")
                print(f"--> New Best Model! Reward: {running_reward:.2f}")
                
            # Чекпоинт
            if episode % self.cfg.checkpoint_every == 0:
                self.save_model("last.pt")
                
        self.save_model("last.pt")
        print("Training finished.")
        return {}

    def run_episode(self) -> tuple[float, int]:
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            self.agent.store_reward(reward)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps >= self.cfg.max_steps_per_episode:
                done = True

        return total_reward, steps

    def save_model(self, name: str) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, name)
        self.agent.save(path)