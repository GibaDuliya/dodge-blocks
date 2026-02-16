import torch
import numpy as np
from torch.distributions import Categorical
from src.agent.policy_network import PolicyNetwork 

class ReinforceAgent:
    def __init__(self, config) -> None:
        self.gamma = config.gamma
        self.lr = config.learning_rate
        self.entropy_coef = config.entropy_coef
        
        self.device = torch.device("cpu") # Для такой маленькой сети CPU часто быстрее
        
        self.policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        probs = self.policy(state_t)
        
        dist = Categorical(probs)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        
        return int(action.item())

    def store_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def update_policy(self) -> float:
        if len(self.rewards) < 2:
            self.clear_buffers()
            return 0.0

        # 1. Расчет дисконтированных вознаграждений
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 2. Стандартизация Returns (КРИТИЧНО)
        # Это заставляет лосс "замечать" разницу между плохим и хорошим эпизодом
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(self.log_probs).squeeze()
        entropies = torch.stack(self.entropies).squeeze()

        # 3. Policy Loss + Entropy Bonus
        # Отрицательный знак, так как мы хотим МАКСИМИЗИРОВАТЬ ожидаемую награду
        policy_loss = -(log_probs * returns).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()
        
        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        # Ограничиваем градиенты, чтобы не было резких скачков
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        val = loss.item()
        self.clear_buffers()
        return val

    def clear_buffers(self):
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))