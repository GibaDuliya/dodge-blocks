import torch
import numpy as np
from torch.distributions import Categorical

from src.agent.policy_network import PolicyNetwork #


class ReinforceAgent:
    """
    Агент, использующий алгоритм REINFORCE (Monte-Carlo Policy Gradient).
    """

    def __init__(self, config: "AgentConfig") -> None:
        """
        Создаёт PolicyNetwork, оптимизатор (Adam).
        Инициализирует буферы log_probs и rewards.
        """
        self.gamma: float = float(getattr(config, "gamma", 0.99))
        self.device = torch.device(getattr(config, "device", "cpu")) # костыль

        state_dim: int = int(getattr(config, "state_dim", 4))
        hidden_dim: int = int(getattr(config, "hidden_dim", 128))
        action_dim: int = int(getattr(config, "action_dim", 3))
        lr: float = float(getattr(config, "learning_rate", 1e-3)) #

        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # буферы эпизода
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

    # ---- взаимодействие со средой ----

    def select_action(self, state: np.ndarray) -> int:
        """
        Прогоняет state через сеть, сэмплирует действие из Categorical,
        сохраняет log_prob в буфер. Возвращает int action.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, state_dim)
        probs = self.policy(state_t)                    # (1, action_dim)
        dist = Categorical(probs)
        action = dist.sample()                          # scalar tensor
        self.log_probs.append(dist.log_prob(action))    # scalar tensor, в графе
        return int(action.item())

    def store_reward(self, reward: float) -> None:
        """Добавляет reward в буфер rewards."""
        self.rewards.append(float(reward))

    # ---- обучение ----

    def compute_returns(self) -> torch.Tensor:
        """
        По буферу rewards считает дисконтированные G_t (от конца к началу).
        Нормализует (zero-mean, unit-std).
        Возвращает tensor returns длины T.
        """
        returns: list[float] = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # нормализация (если > 1 шага и std != 0) # нужна ли ??
        # if returns_t.numel() > 1:
        #     std = returns_t.std(unbiased=False)
        #     if std.item() > 1e-8:
        #         returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)
        return returns_t

    def update_policy(self) -> float:
        """
        Вычисляет REINFORCE loss = −Σ log_prob_t · G_t,
        делает backward + optimizer step.
        Очищает буферы (вызывает clear_buffers).
        Возвращает float loss для логирования.
        """
        returns_t = self.compute_returns()                     # (T,)
        log_probs_t = torch.stack(self.log_probs)              # (T,)

        loss = -(log_probs_t * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_value = loss.item()
        self.clear_buffers()
        return loss_value

    def clear_buffers(self) -> None:
        """Очищает списки log_probs и rewards."""
        self.log_probs.clear()
        self.rewards.clear()

    # ---- сериализация ----

    def save(self, path: str) -> None:
        """torch.save state_dict + optimizer state_dict."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """torch.load и загружает state_dict'ы."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])