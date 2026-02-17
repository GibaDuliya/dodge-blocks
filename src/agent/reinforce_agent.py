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
        self.grid_height = 12  # Will be set by trainer
        self.state_mode = "absolute"  # Will be set by trainer
        
        self.device = torch.device("cpu")
        self.policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.heights = []  # Track block_y at each step
        
        # Baseline statistics - rolling window of last 50 episodes
        self.episode_outcomes = deque(maxlen=50)  # Store 'miss' or 'death' for each episode

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        
        # Extract block_y from state based on state_mode
        if self.state_mode == "relative":
            # state[1] is block_y / grid_height, denormalize it
            block_y = state[1] * self.grid_height
        else:
            # state[3] is the raw block_y
            block_y = state[3]
        
        self.heights.append(block_y)
        return int(action.item())

    def store_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def update_episode_stats(self, info: dict) -> None:
        """Update miss/death counts from episode info (rolling window of 50 episodes)."""
        if info.get('miss', False):
            self.episode_outcomes.append('miss')
        elif info.get('death', False):
            self.episode_outcomes.append('death')

    def compute_p_miss(self) -> float:
        """Compute probability of surviving a landing from last 50 episodes."""
        if len(self.episode_outcomes) == 0:
            return 0.5  # Safe default
        
        miss_count = sum(1 for outcome in self.episode_outcomes if outcome == 'miss')
        total = len(self.episode_outcomes)
        
        if total == 0:
            return 0.5
        return miss_count / total

    def compute_value_baseline(self, heights: torch.Tensor) -> torch.Tensor:
        """
        Compute analytic value baseline V(h) for given heights.
        
        V(0) = (11*p_miss - 10) / (1 - p_miss * gamma^(H+1))
        V(h) = gamma^h * V(0)
        
        Where H = grid_height - 1, r_miss=1, r_death=-10
        """
        p_miss = self.compute_p_miss()
        
        # Compute V(0)
        H = self.grid_height - 1
        denominator = 1.0 - p_miss * (self.gamma ** (H + 1))
        
        # Avoid division by zero
        if abs(denominator) < 1e-8:
            V0 = 0.0
        else:
            numerator = 11 * p_miss - 10
            V0 = numerator / denominator
        
        # Compute V(h) = gamma^h * V(0)
        # heights are float tensors
        V_h = (self.gamma ** heights) * V0
        return V_h

    def update_policy(self) -> float:
        if len(self.rewards) < 2:
            self.clear_buffers()
            return 0.0

        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Apply baseline if enabled
        if self.use_height_baseline and len(self.heights) > 0:
            # Ensure heights tensor matches returns length
            heights_t = torch.tensor(self.heights, dtype=torch.float32, device=self.device)
            if len(heights_t) != len(returns):
                # Handle mismatch (shouldn't happen in normal operation)
                heights_t = heights_t[:len(returns)]
            baselines = self.compute_value_baseline(heights_t)
            advantages = returns - baselines
        else:
            advantages = returns
        
        # 1. Ablation: Normalization
        if self.use_norm and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(self.log_probs).squeeze()
        entropies = torch.stack(self.entropies).squeeze()

        policy_loss = -(log_probs * advantages).mean()
        
        # 2. Ablation: Entropy Bonus
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