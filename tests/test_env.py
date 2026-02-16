# tests/test_env.py
from dataclasses import dataclass

import numpy as np
import pytest

from src.environment.game_env import GameEnv


@dataclass
class DummyEnvConfig:
    grid_width: int = 7
    grid_height: int = 6
    block_min_width: int = 1
    block_max_width: int = 1
    block_fall_speed: int = 1
    seed: int = 123
    agent_start_x: int = 3


class TestGameEnv:
    def test_reset_returns_valid_state(self):
        cfg = DummyEnvConfig()
        env = GameEnv(cfg)

        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)

        agent_x, block_left, block_right, block_y = state.tolist()

        assert 0 <= agent_x <= cfg.grid_width - 1
        assert 0 <= block_left <= block_right <= cfg.grid_width - 1
        assert block_y == cfg.grid_height - 1  # spawned at the top

    def test_step_returns_correct_tuple(self):
        cfg = DummyEnvConfig()
        env = GameEnv(cfg)
        env.reset()

        out = env.step(1)  # stay
        assert isinstance(out, tuple)
        assert len(out) == 4

        next_state, reward, done, info = out
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "survived_steps" in info
        assert isinstance(info["survived_steps"], int)

    def test_agent_clipped_to_boundaries(self):
        cfg = DummyEnvConfig(grid_width=5, agent_start_x=0)
        env = GameEnv(cfg)
        env.reset()

        # Try to go left at x=0
        env.agent_x = 0
        env.step(0)
        assert env.agent_x == 0

        # Try to go right at x=width-1
        env.agent_x = cfg.grid_width - 1
        env.step(2)
        assert env.agent_x == cfg.grid_width - 1

    def test_collision_detected(self):
        cfg = DummyEnvConfig(grid_width=7, agent_start_x=3, block_fall_speed=1)
        env = GameEnv(cfg)
        env.reset()

        # Place a block one step above the agent so it lands this step
        env.agent_x = 3
        env.block_left = 2
        env.block_right = 4  # covers agent_x=3
        env.block_y = 1

        next_state, reward, done, info = env.step(1)  # stay
        assert done is True
        assert reward == -10.0

    def test_miss_gives_positive_reward(self):
        cfg = DummyEnvConfig(grid_width=7, agent_start_x=3, block_fall_speed=1)
        env = GameEnv(cfg)
        env.reset()

        # Block lands this step but misses agent
        env.agent_x = 3
        env.block_left = 0
        env.block_right = 1  # does NOT cover agent_x=3
        env.block_y = 1

        next_state, reward, done, info = env.step(1)  # stay
        assert done is False
        assert reward == 1.0
        assert info["survived_steps"] == 1
        # After a miss, a new block should be spawned at the top
        assert env.block_y == cfg.grid_height - 1


if __name__ == "__main__":
    pytest.main([__file__])
