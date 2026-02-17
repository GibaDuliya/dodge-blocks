import torch
import numpy as np
from torch.distributions import Categorical
from collections import deque
from src.agent.policy_network import PolicyNetwork 

class ReinforceAgent:
    def __init__(self, config) -> None:
        self.gamma = config.gamma
        self.lr = config.learning_rate
        self.entropy_coef = config.entropy_coef
        self.use_norm = config.use_normalization
        self.use_height_baseline = config.use_height_baseline
        
        # Настройки среды (уточняются тренером)
        self.grid_height = 12  
        self.state_mode = "absolute"
        
        self.device = torch.device("cpu")
        self.policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # Буферы эпизода
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.heights = []
        
        # --- Адаптивный Baseline ---
        self.episode_outcomes = []  # Список для хранения исходов
        self.episodes_count = 0     # Общий счетчик для изменения окна
        self.max_window = 100       # Начальное (большое) окно для стабильности
        self.min_window = 15        # Конечное (узкое) окно для скорости реакции
        self.decay_steps = 1000     # За сколько эпизодов окно сузится до минимума

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        
        # Извлечение block_y
        if self.state_mode == "relative":
            block_y = state[1] * self.grid_height
        else:
            block_y = state[3]
        
        self.heights.append(block_y)
        return int(action.item())

    def store_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def update_episode_stats(self, info: dict) -> None:
        """Обновляет исходы с использованием адаптивного размера окна."""
        outcome = None
        if info.get('miss', False):
            outcome = 'miss'
        elif info.get('death', False):
            outcome = 'death'
        
        if outcome:
            self.episode_outcomes.append(outcome)
            self.episodes_count += 1
            
            # Линейное уменьшение размера окна от max до min
            # Чем больше episodes_count, тем меньше current_max_len
            fraction = min(self.episodes_count / self.decay_steps, 1.0)
            current_max_len = int(self.max_window - (self.max_window - self.min_window) * fraction)
            
            # Обрезаем список исходов до текущего размера окна
            if len(self.episode_outcomes) > current_max_len:
                self.episode_outcomes = self.episode_outcomes[-current_max_len:]

    def compute_p_miss(self) -> float:
        """Вычисляет P(miss) на основе текущего (адаптивного) окна."""
        if not self.episode_outcomes:
            return 0.5
        
        miss_count = sum(1 for outcome in self.episode_outcomes if outcome == 'miss')
        return miss_count / len(self.episode_outcomes)

    def compute_value_baseline(self, heights: torch.Tensor) -> torch.Tensor:
        """
        Аналитический baseline V(h).
        V(0) = (11*p_miss - 10) / (1 - p_miss * gamma^(H+1))
        V(h) = gamma^h * V(0)
        """
        p_miss = self.compute_p_miss()
        H = self.grid_height - 1
        
        denominator = 1.0 - p_miss * (self.gamma ** (H + 1))
        
        if abs(denominator) < 1e-8:
            V0 = 0.0
        else:
            # Награды: r_miss = 1, r_death = -10. Итого: 1*p + (-10)*(1-p) = 11p - 10
            numerator = 11 * p_miss - 10
            V0 = numerator / denominator
        
        V_h = (self.gamma ** heights) * V0
        return V_h

    def update_policy(self) -> float:
        if len(self.rewards) < 2:
            self.clear_buffers()
            return 0.0

        # Расчет дисконтированных вознаграждений (G_t)
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Применение Baseline
        if self.use_height_baseline and len(self.heights) > 0:
            heights_t = torch.tensor(self.heights, dtype=torch.float32, device=self.device)
            # Синхронизация длин (на случай преждевременного конца эпизода)
            if len(heights_t) != len(returns):
                heights_t = heights_t[:len(returns)]
                
            baselines = self.compute_value_baseline(heights_t)
            advantages = returns - baselines
        else:
            advantages = returns
        
        # Нормализация преимуществ
        if self.use_norm and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(self.log_probs).squeeze()
        entropies = torch.stack(self.entropies).squeeze()

        # Loss = Policy Loss + Entropy Bonus
        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()
        
        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        val = loss.item()
        self.clear_buffers()
        return val

    def clear_buffers(self):
        self.log_probs, self.rewards, self.entropies, self.heights = [], [], [], []

    def save(self, path): torch.save(self.policy.state_dict(), path)
    def load(self, path): self.policy.load_state_dict(torch.load(path, map_location=self.device))