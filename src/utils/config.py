from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class EnvConfig:
    grid_width: int = 6
    grid_height: int = 12
    block_min_width: int = 1
    block_max_width: int = 2
    block_fall_speed: int = 1

@dataclass
class AgentConfig:
    state_dim: int = 4
    hidden_dim: int = 128
    action_dim: int = 3
    learning_rate: float = 0.001 
    gamma: float = 0.99
    entropy_coef: float = 0.01   # Уменьшили энтропию, чтобы он быстрее фокусировался на цели

@dataclass
class TrainConfig:
    num_episodes: int = 500
    max_steps_per_episode: int = 2000
    checkpoint_every: int = 400
    log_every: int = 50
    stats_path: str = "artifacts/logs/train_stats.csv"
    checkpoint_dir: str = "artifacts/checkpoints"
# ... RenderConfig без изменений
@dataclass
class RenderConfig:
    cell_size: int = 40
    fps: int = 30
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "agent": (50, 200, 50),
        "block": (200, 50, 50),
        "bg": (30, 30, 30),
        "grid": (50, 50, 50),
        "text": (255, 255, 255)
    })