import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Простая полносвязная сеть: state → softmax → вероятности 3 действий.
    Архитектура: Linear → ReLU → Linear → ReLU → Linear → Softmax.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        """Определяет слои."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Принимает батч состояний shape (B, state_dim).
        Возвращает вероятности действий shape (B, action_dim).
        """
        return self.net(x)