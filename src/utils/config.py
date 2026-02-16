from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    grid_width: int = 5          # ширина поля (число клеток)
    block_min_width: int = 1   # мин. ширина блока
    block_max_width: int = 1      # макс. ширина блока
    block_fall_speed: int = 1    # на сколько клеток блок опускается за шаг
    grid_height: int = 4         # высота поля (откуда блок начинает падать)
    agent_start_pos: int = None     # стартовая позиция агента (None → центр)

@dataclass
class AgentConfig:
    state_dim: int           # размерность входа (= 4)
    hidden_dim: int          # нейроны скрытого слоя
    action_dim: int          # размерность выхода (= 3)
    learning_rate: float
    gamma: float             # дисконт-фактор

@dataclass
class TrainConfig:
    num_episodes: int
    max_steps_per_episode: int
    checkpoint_every: int     # сохранять модель каждые N эпизодов
    log_every: int            # логировать каждые N эпизодов
    stats_path: str           # путь к CSV
    checkpoint_dir: str       # папка чекпоинтов

@dataclass
class RenderConfig:
    cell_size: int            # размер клетки в пикселях
    fps: int                  # кадров в секунду
    colors: dict              # словарь цветов {"agent": ..., "block": ..., "bg": ...}