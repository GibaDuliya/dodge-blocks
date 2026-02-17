import numpy as np

class GameEnv:
    def __init__(self, config, seed) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent_x = self.cfg.grid_width // 2
        self.done = False
        self._spawn_block()
        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done: return self.get_state(), 0.0, True, {}

        if action == 0: self.agent_x = max(0, self.agent_x - 1)
        elif action == 2: self.agent_x = min(self.cfg.grid_width - 1, self.agent_x + 1)
        
        self.block_y -= self.cfg.block_fall_speed

        # 4. Ablation: Reward Math
        reward = 0.0
        collision = (self.block_y <= 0 and (self.block_left <= self.agent_x <= self.block_right))
        passed = (self.block_y < 0)
        info = {}

        if self.cfg.reward_mode == "enhanced":
            reward = 0.1 
            if collision:
                reward = -15.0
                self.done = True
                info['death'] = True
            elif passed:
                reward = 10.0
                self._spawn_block()
                info['miss'] = True
        else: # Basic mode (старый вариант)
            if collision:
                reward = -10.0
                self.done = True
                info['death'] = True
            elif passed:
                reward = 1.0
                self._spawn_block()
                info['miss'] = True

        return self.get_state(), reward, self.done, info

    def get_state(self) -> np.ndarray:
        # 3. Ablation: State Representation
        if self.cfg.state_mode == "relative":
            gw, gh = self.cfg.grid_width, self.cfg.grid_height
            dist_left = (self.agent_x - self.block_left) / gw
            dist_right = (self.agent_x - self.block_right) / gw
            return np.array([self.agent_x/(gw-1), self.block_y/gh, dist_left, dist_right], dtype=np.float32)
        else:
            # Absolute (старый вариант)
            return np.array([self.agent_x, self.block_left, self.block_right, self.block_y], dtype=np.float32)

    def _spawn_block(self) -> None:
        w = self.rng.integers(self.cfg.block_min_width, self.cfg.block_max_width + 1)
        self.block_left = self.rng.integers(0, self.cfg.grid_width - w + 1)
        self.block_right = self.block_left + w - 1
        self.block_y = self.cfg.grid_height