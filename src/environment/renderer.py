import pygame
import numpy as np
from src.utils.config import EnvConfig, RenderConfig

class GameRenderer:
    """Реализует Pygame-отрисовку состояния GameEnv."""

    def __init__(self, env_config: EnvConfig, render_config: RenderConfig) -> None:
        self.env_cfg = env_config
        self.render_cfg = render_config
        self.screen = None
        self.clock = None
        self.font = None

        # Размеры окна
        self.win_width = self.env_cfg.grid_width * self.render_cfg.cell_size
        self.win_height = self.env_cfg.grid_height * self.render_cfg.cell_size

    def init_display(self) -> None:
        pygame.init()
        # Используем встроенный шрифт Pygame вместо системного Arial
        # Это гарантированно лечит ошибку на macOS/Python 3.14
        try:
            pygame.font.init()
            self.font = pygame.font.Font(None, 36) # None — это встроенный шрифт
        except:
            self.font = None
            print("Font still not working")

        self.screen = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption("Dodge Blocks")
        self.clock = pygame.time.Clock()

    def render(self, state: np.ndarray, score: int) -> None:
        if self.screen is None:
            self.init_display()

        self.screen.fill(self.render_cfg.colors["bg"])
        c_size = self.render_cfg.cell_size

        # Рисуем сетку
        for x in range(0, self.win_width, c_size):
            pygame.draw.line(self.screen, self.render_cfg.colors["grid"], (x, 0), (x, self.win_height))
        for y in range(0, self.win_height, c_size):
            pygame.draw.line(self.screen, self.render_cfg.colors["grid"], (0, y), (self.win_width, y))

        # ТВОЙ ФОРМАТ: [agent_x, block_left, block_right, block_y]
        agent_x, b_left, b_right, b_y = state
        
        # Инвертируем Y для отрисовки (в Pygame 0 — это верх, а в твоем env 0 — это низ)
        display_y = (self.env_cfg.grid_height - 1 - b_y) * c_size
        agent_display_y = (self.env_cfg.grid_height - 1) * c_size

        # Рисуем Агента
        agent_rect = pygame.Rect(agent_x * c_size, agent_display_y, c_size, c_size)
        pygame.draw.rect(self.screen, self.render_cfg.colors["agent"], agent_rect)

        # Рисуем Блок
        block_width_cells = (b_right - b_left) + 1
        block_rect = pygame.Rect(b_left * c_size, display_y, block_width_cells * c_size, c_size)
        pygame.draw.rect(self.screen, self.render_cfg.colors["block"], block_rect)

        # Счет (безопасная отрисовка только если шрифт доступен)
        if self.font:
            try:
                text_surface = self.font.render(f"Score: {score}", True, self.render_cfg.colors["text"])
                self.screen.blit(text_surface, (10, 10))
            except:
                pass

        pygame.display.flip()
        self.clock.tick(self.render_cfg.fps)


    def render_menu(self, score: int) -> None:
        """Отрисовывает оверлей меню поверх последнего кадра."""
        if not self.font: return

        # Затемнение экрана
        overlay = pygame.Surface((self.win_width, self.win_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) # Полупрозрачный черный
        self.screen.blit(overlay, (0, 0))

        # Тексты
        texts = [
            f"GAME OVER",
            f"Final Score: {score}",
            "",
            "Press [R] to Restart",
            "Press [Q] to Quit"
        ]

        for i, line in enumerate(texts):
            color = (255, 255, 255) if i != 0 else (200, 50, 50)
            surf = self.font.render(line, True, color)
            rect = surf.get_rect(center=(self.win_width // 2, self.win_height // 3 + i * 40))
            self.screen.blit(surf, rect)

        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def handle_events(self) -> bool:
        """
        Обрабатывает очередь событий. 
        Убрали @staticmethod, чтобы иметь доступ к self.init_display()
        """
        if self.screen is None:
            self.init_display()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True