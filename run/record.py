import pygame
import torch
import sys
import os
import imageio
import numpy as np

sys.path.append(os.getcwd())

from src.environment.game_env import GameEnv
from src.environment.renderer import GameRenderer
from src.agent.reinforce_agent import ReinforceAgent
from src.utils.config import EnvConfig, RenderConfig, AgentConfig


def capture_frame(renderer):
    frame = pygame.surfarray.array3d(renderer.screen)
    return np.transpose(frame, (1, 0, 2))


def main(number):
    env = GameEnv(EnvConfig(), seed=number)
    renderer = GameRenderer(EnvConfig(), RenderConfig())
    agent = ReinforceAgent(AgentConfig())

    model_path = "artifacts/checkpoints/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    agent.load(model_path)
    print(f"Loaded model: {model_path}")

    state = env.reset()
    frames = []
    total_score = 0

    for step in range(2000):
        if not renderer.handle_events():
            break

        with torch.no_grad():
            action = agent.select_action(state)
            agent.clear_buffers()

        state, reward, done, _ = env.step(action)
        total_score += reward
        renderer.render(env, int(total_score))
        frames.append(capture_frame(renderer))

        if done:
            break

    renderer.close()

    if frames:
        gif_path = f"analysis/records/gameplay_{number}.gif"
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        imageio.mimsave(gif_path, frames, duration=1 / 15, loop=0)
        print(f"GIF saved: {gif_path} | {len(frames)} frames | score: {int(total_score)}")


if __name__ == "__main__":
    for i in range(10):
        main(i)