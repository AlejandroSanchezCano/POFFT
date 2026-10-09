"""
===============================================================================
Title:      Epoch Result
Outline:    EpochResult class to store the results of a training or evaluation
            epoch. It provides methods to:
            - Convert the results to a pandas DataFrame.
            - Compute the optimal threshold for binary classification based on
              the Matthews correlation coefficient (MCC).
Author:     Alejandro Sánchez Cano
Date:       09/10/2026
===============================================================================
"""

# Built-in modules
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party modules
import numpy as np
import pandas as pd

# Custom modules
from core.performance import Performance

class EpochResult(ABC):
    pass
    
@dataclass(frozen=True, slots=True)
class EpochResultSTL(EpochResult):
    loss: float
    logits: np.ndarray
    labels: np.ndarray
    identifiers: list

    def to_dataframe(self) -> pd.DataFrame:
        '''Convert the epoch result to a pandas DataFrame.'''
        return pd.DataFrame({
            'logits': self.logits,
            'labels': self.labels,
            'identifiers': self.identifiers
        })

    @property
    def optimal_threshold(self) -> float:
        '''
        Compute the optimal threshold for binary classification based on the 
        MCC score.
        
        Returns
        -------
        float
            Optimal threshold value.
        '''
        # Initialize best values
        best_mcc = -1
        best_threshold = None

        # Iterate over thresholds to find the one that maximizes MCC
        for threshold in np.linspace(0.01, 1, 100):
            performance = Performance(
                true=self.labels,
                logits=self.logits,
                threshold=threshold
            )
            if performance.mcc > best_mcc:
                best_mcc = performance.mcc
                best_threshold = threshold

        return best_threshold

@dataclass(frozen=True, slots=True)
class EpochResultMTL(EpochResult):
    weights: np.ndarray
    losses: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    identifiers: list[list]

if __name__ == "__main__":
    epoch_result = EpochResultSTL(
        loss=0.5,
        logits=np.array([0.1, 0.2, 0.3]),
        labels=np.array([0, 1, 0]),
        identifiers=['id1', 'id2', 'id3']
    )
    print(epoch_result.to_dataframe())
    print("Optimal threshold:", epoch_result.optimal_threshold)