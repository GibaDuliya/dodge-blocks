from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

@dataclass
class EnvConfig:
    grid_width: int = 6
    grid_height: int = 12
    block_min_width: int = 1
    block_max_width: int = 2
    block_fall_speed: int = 1
    # Ablation Flags
    state_mode: str = "absolute" # "absolute" или "relative"
    reward_mode: str = "basic" # "basic" или "enhanced"

@dataclass
class AgentConfig:
    state_dim: int = 4
    hidden_dim: int = 128
    action_dim: int = 3
    learning_rate: float = 0.001 
    gamma: float = 0.99
    # Ablation Flags
    use_normalization: bool = True
    entropy_coef: float = 0.0 # 0.0 для отключения
    use_height_baseline: bool = False

@dataclass
class TrainConfig:
    num_episodes: int = 800
    max_steps_per_episode: int = 2000
    checkpoint_every: int = 500
    log_every: int = 1
    # Пути будут динамическими в зависимости от эксперимента
    exp_name: str = "default"
    stats_path: str = "" 
    checkpoint_dir: str = ""
    early_stop_window: int = 50
    early_stop_threshold: float = 0.8

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