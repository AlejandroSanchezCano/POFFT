"""
===============================================================================
Title:      Seed
Outline:    Functions to set random seeds for deep learning models to ensure
            reproducibility.
Author:     Alejandro Sánchez Cano
Date:       23/09/2026
===============================================================================
"""

# Built-in modules
import random

# Third-party modules
import torch
import numpy as np

def set_seed(seed: int) -> None:
    '''
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed to set.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    '''
    Worker initialization function for reproducibility in data loading.

    Parameters
    ----------
    worker_id : int
        Worker ID for the data loader.
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)