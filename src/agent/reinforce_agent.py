import torch
import numpy as np

class ReinforceAgent:
    """
    Агент, использующий алгоритм REINFORCE (Monte-Carlo Policy Gradient).
    """

    def __init__(self, config: "AgentConfig") -> None:
        """
        Создаёт PolicyNetwork, оптимизатор (Adam).
        Инициализирует буферы log_probs и rewards.
        """
        ...

    # ---- взаимодействие со средой ----

    def select_action(self, state: np.ndarray) -> int:
        """
        Прогоняет state через сеть, сэмплирует действие из Categorical,
        сохраняет log_prob в буфер. Возвращает int action.
        """
        ...

    def store_reward(self, reward: float) -> None:
        """Добавляет reward в буфер rewards."""
        ...

    # ---- обучение ----

    def compute_returns(self) -> torch.Tensor:
        """
        По буферу rewards считает дисконтированные G_t.
        Нормализует (zero-mean, unit-std).
        Возвращает tensor returns.
        """
        ...

    def update_policy(self) -> float:
        """
        Вычисляет REINFORCE loss = −Σ log_prob * G_t,
        делает backward + optimizer step.
        Очищает буферы (вызывает clear_buffers).
        Возвращает float loss для логирования.
        """
        ...

    def clear_buffers(self) -> None:
        """Очищает списки log_probs и rewards."""
        ...

    # ---- сериализация ----

    def save(self, path: str) -> None:
        """torch.save state_dict + optimizer state_dict."""
        ...

    def load(self, path: str) -> None:
        """torch.load и загружает state_dict'ы."""
        ...