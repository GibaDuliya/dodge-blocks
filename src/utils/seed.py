import random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
<<<<<<< HEAD
    """Фиксирует seed для random, numpy, torch (кроме game_env.py в np.random.default_rng)"""
    import random
    import numpy as np
    import torch

=======
    """Фиксирует seed для random, numpy, torch."""
>>>>>>> 0af4050cd535a8385e10ea7374bb0fcfcb875135
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)