import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    Простая полносвязная сеть: state → softmax → вероятности 3 действий.
    Архитектура: Linear → ReLU → Linear → ReLU → Linear → Softmax.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        """Определяет слои."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Принимает батч состояний shape (B, state_dim).
        Возвращает вероятности действий shape (B, action_dim).
        """
        ...