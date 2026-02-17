import random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    """Фиксирует seed для random, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)