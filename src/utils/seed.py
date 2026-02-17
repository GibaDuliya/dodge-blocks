def set_global_seed(seed: int) -> None:
    """Фиксирует seed для random, numpy, torch (кроме game_env.py в np.random.default_rng)"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)