"""
===============================================================================
Title:      Utils
Outline:    Utility functions for the training scripts:
            - extract_pairs: gets protein pairs from a subset of a dataset.
Author:     Alejandro Sánchez Cano
Date:       08/10/2026
===============================================================================
"""

# Third-party modules
import torch

def extract_pairs(
    subset: torch.utils.data.dataset.Subset
) -> list['ProteinPair']:
    '''
    Extracts the protein pairs from a subset of a dataset.

    Parameters
    ----------
    subset : torch.utils.data.dataset.Subset
        A subset of a dataset containing protein pairs.

    Returns
    -------
    list['ProteinPair']
        A list of protein pairs extracted from the subset.
    '''
    return [subset.dataset.pairs[idx] for idx in subset.indices]