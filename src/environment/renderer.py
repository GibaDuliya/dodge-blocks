import pygame
import numpy as np
from src.utils.config import EnvConfig, RenderConfig

class GameRenderer:
    def __init__(self, env_config: EnvConfig, render_config: RenderConfig) -> None:
        self.env_cfg = env_config
        self.render_cfg = render_config
        self.screen = None
        self.clock = None
        self.font = None

        self.win_width = self.env_cfg.grid_width * self.render_cfg.cell_size
        self.win_height = self.env_cfg.grid_height * self.render_cfg.cell_size

    def init_display(self) -> None:
        pygame.init()
        try:
            pygame.font.init()
            self.font = pygame.font.Font(None, 36)
        except:
            self.font = None

        self.screen = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption("Dodge Blocks - RL Agent")
        self.clock = pygame.time.Clock()

    def render(self, env, score: int) -> None:
        """
        Берем данные напрямую из env, а не из нормализованного state.
        """
        if self.screen is None:
            self.init_display()

        self.screen.fill(self.render_cfg.colors["bg"])
        c_size = self.render_cfg.cell_size

        # Сетка
        for x in range(0, self.win_width, c_size):
            pygame.draw.line(self.screen, self.render_cfg.colors["grid"], (x, 0), (x, self.win_height))
        for y in range(0, self.win_height, c_size):
            pygame.draw.line(self.screen, self.render_cfg.colors["grid"], (0, y), (self.win_width, y))

        # Координаты из среды
        agent_x = env.agent_x
        b_left = env.block_left
        b_right = env.block_right
        b_y = env.block_y
        
        # Инверсия Y (в Pygame 0 - это верх)
        display_y = (self.env_cfg.grid_height - 1 - b_y) * c_size
        agent_display_y = (self.env_cfg.grid_height - 1) * c_size

        # Рисуем Агента
        agent_rect = pygame.Rect(int(agent_x * c_size), int(agent_display_y), c_size, c_size)
        pygame.draw.rect(self.screen, self.render_cfg.colors["agent"], agent_rect)

        # Рисуем Блок
        block_width_cells = (b_right - b_left) + 1
        block_rect = pygame.Rect(int(b_left * c_size), int(display_y), int(block_width_cells * c_size), c_size)
        pygame.draw.rect(self.screen, self.render_cfg.colors["block"], block_rect)

        # Счет
        if self.font:
            text_surface = self.font.render(f"Score: {score}", True, self.render_cfg.colors["text"])
            self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.render_cfg.fps)

    def render_menu(self, score: int) -> None:
        if not self.font: return
        overlay = pygame.Surface((self.win_width, self.win_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) 
        self.screen.blit(overlay, (0, 0))

        texts = ["GAME OVER", f"Score: {score}", "", "[R] Restart", "[Q] Quit"]
        for i, line in enumerate(texts):
            color = (255, 255, 255) if i != 0 else (200, 50, 50)
            surf = self.font.render(line, True, color)
            rect = surf.get_rect(center=(self.win_width // 2, self.win_height // 3 + i * 40))
            self.screen.blit(surf, rect)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def handle_events(self) -> bool:
        if self.screen is None: self.init_display()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
        return True