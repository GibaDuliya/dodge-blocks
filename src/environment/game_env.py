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
        self.config = config

        # Ожидаемые поля в конфиге (можно хранить в dataclass EnvConfig)
        self.grid_width: int = int(getattr(config, "grid_width"))
        self.grid_height: int = int(getattr(config, "grid_height"))
        self.block_min_width: int = int(getattr(config, "block_min_width", 1))
        self.block_max_width: int = int(getattr(config, "block_max_width", 1))
        self.block_fall_speed: int = int(getattr(config, "block_fall_speed", 1))

        seed = getattr(config, "seed", None) # Возможно надо самому поставить seed 
        self.rng = np.random.default_rng(seed)

        # internal state
        self.agent_x: int = 0
        self.block_left: int = 0
        self.block_right: int = 0
        self.block_y: int = 0
        self.survived_steps: int = 0
        self.done: bool = False

        self.reset()

    # ---- публичный API ----

    def reset(self) -> np.ndarray:
        """Сбрасывает состояние среды, возвращает начальный state."""
        # агент внизу, стартуем по центру (или из конфига)
        start_x = getattr(self.config, "agent_start_x", self.grid_width // 2)
        self.agent_x = int(np.clip(start_x, 0, self.grid_width - 1))

        self.survived_steps = 0
        self.done = False

        self._spawn_block()
        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Принимает action, возвращает (next_state, reward, done, info).
        info может содержать {"survived_steps": int}.
        """
        if self.done:
            # если эпизод завершён — возвращаем текущее состояние без изменений
            return self.get_state(), 0.0, True, {"survived_steps": self.survived_steps}  # reward -10?

        # 1) применяем действие агента
        self._move_agent(action)

        # 2) двигаем блок
        self._update_block()

        # 3) считаем награду/терминацию
        reward = 0.0

        if self._check_collision():
            reward = -10.0
            self.done = True
        elif self._block_landed():
            # блок долетел до низа и НЕ попал в агента
            reward = 1.0
            self._spawn_block()

        # 4) шаг выживания (каждый step, пока done=False)
        if not self.done:
            self.survived_steps += 1

        return self.get_state(), float(reward), bool(self.done), {"survived_steps": self.survived_steps}

    def get_state(self) -> np.ndarray:
        """Возвращает текущий state как np.array shape (4,)."""
        return np.array(
            [self.agent_x, self.block_left, self.block_right, self.block_y],
            dtype=np.int32,
        )

    # ---- внутренние helpers ----

    def _spawn_block(self) -> None:
        """Генерирует новый блок случайной ширины наверху поля."""
        w_min = max(1, self.block_min_width)
        w_max = max(w_min, self.block_max_width)
        width = int(self.rng.integers(w_min, w_max + 1))

        # выбираем левую границу так, чтобы блок влезал в [0, grid_width-1]
        max_left = max(0, self.grid_width - width)
        left = int(self.rng.integers(0, max_left + 1))
        right = left + width - 1

        self.block_left = left
        self.block_right = right
        # стартуем сверху (y = grid_height - 1), агент на y = 0
        self.block_y = self.grid_height - 1

    def _move_agent(self, action: int) -> None:
        """Двигает агента на ±1 или оставляет на месте; клиппит границы."""
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}. Expected one of (0, 1, 2).")

        dx = -1 if action == 0 else (1 if action == 2 else 0)
        self.agent_x = int(np.clip(self.agent_x + dx, 0, self.grid_width - 1))

    def _update_block(self) -> None:
        """Опускает текущий блок на block_fall_speed клеток."""
        speed = max(1, int(self.block_fall_speed))
        self.block_y -= speed

    def _check_collision(self) -> bool:
        """True если блок достиг уровня агента И перекрывает его координату."""
        # агент находится на уровне y = 0
        if self.block_y > 0:
            return False
        return self.block_left <= self.agent_x <= self.block_right

    def _block_landed(self) -> bool:
        """True если блок достиг нижней границы (y <= 0)."""
        return self.block_y <= 0
