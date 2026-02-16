import numpy as np
from src.utils.config import EnvConfig

class GameEnv:
    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.grid_width = config.grid_width
        self.grid_height = config.grid_height
        self.block_fall_speed = config.block_fall_speed
        
        self.rng = np.random.default_rng()
        
        self.agent_x = 0
        self.block_left = 0
        self.block_right = 0
        self.block_y = 0
        self.done = False
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent_x = self.grid_width // 2
        self.done = False
        self._spawn_block()
        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self.get_state(), 0.0, True, {}

        # 0=Left, 1=Stay, 2=Right
        if action == 0:
            self.agent_x = max(0, self.agent_x - 1)
        elif action == 2:
            self.agent_x = min(self.grid_width - 1, self.agent_x + 1)
        
        self.block_y -= self.block_fall_speed

        # 1. Маленькая награда за сам факт существования (стимул не умирать сразу)
        reward = 0.1 

        # 2. Проверка столкновения
        if self.block_y <= 0 and (self.block_left <= self.agent_x <= self.block_right):
            reward = -15.0 # ШТРАФ ЗА СМЕРТЬ (сделали больше)
            self.done = True
        
        # 3. Успешный пролет блока (ГЛАВНАЯ ЦЕЛЬ)
        elif self.block_y < 0:
            reward = 10.0  # БОЛЬШОЙ БОНУС за каждый пройденный блок
            self._spawn_block()

        return self.get_state(), reward, self.done, {}

    def get_state(self) -> np.ndarray:
        """Относительные координаты - лучший способ для REINFORCE."""
        gw = self.grid_width
        gh = self.grid_height
        
        # Насколько агент далеко от краев блока
        # Положительное число значит агент правее края, отрицательное - левее
        dist_left = (self.agent_x - self.block_left)
        dist_right = (self.agent_x - self.block_right)
        
        return np.array([
            self.agent_x / (gw - 1),
            self.block_y / gh,
            dist_left / gw,
            dist_right / gw
        ], dtype=np.float32)

    def _spawn_block(self) -> None:
        w_min = max(1, getattr(self.config, 'block_min_width', 1))
        w_max = max(w_min, getattr(self.config, 'block_max_width', 2))
        width = self.rng.integers(w_min, w_max + 1)
        width = min(width, self.grid_width - 2) # Всегда оставляем проход минимум в 2 клетки

        max_left = max(0, self.grid_width - width)
        self.block_left = self.rng.integers(0, max_left + 1)
        self.block_right = self.block_left + width - 1
        self.block_y = self.grid_height