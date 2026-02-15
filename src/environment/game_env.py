import numpy as np

class GameEnv:
    """
    Одномерная сетка. Сверху падают блоки.
    state = (agent_x, block_left, block_right, block_y)
    action ∈ {0: влево, 1: на месте, 2: вправо}
    reward: +1 если блок упал мимо, −10 если попал по агенту.
    """

    def __init__(self, config: "EnvConfig") -> None:
        """Сохраняет конфиг, вызывает reset()."""
        ...

    # ---- публичный API ----

    def reset(self) -> np.ndarray:
        """Сбрасывает состояние среды, возвращает начальный state."""
        ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Принимает action, возвращает (next_state, reward, done, info).
        info может содержать {"survived_steps": int}.
        """
        ...

    def get_state(self) -> np.ndarray:
        """Возвращает текущий state как np.array shape (4,)."""
        ...

    # ---- внутренние helpers ----

    def _spawn_block(self) -> None:
        """Генерирует новый блок случайной ширины наверху поля."""
        ...

    def _move_agent(self, action: int) -> None:
        """Двигает агента на ±1 или оставляет на месте; клиппит границы."""
        ...

    def _update_block(self) -> None:
        """Опускает текущий блок на block_fall_speed клеток."""
        ...

    def _check_collision(self) -> bool:
        """True если блок достиг уровня агента И перекрывает его координату."""
        ...

    def _block_landed(self) -> bool:
        """True если блок достиг нижней границы (y <= 0)."""
        ...