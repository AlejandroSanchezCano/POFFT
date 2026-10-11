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
    logits: list[np.ndarray]
    labels: list[np.ndarray]
    identifiers: list[list]

    @classmethod
    def from_stls(
        cls,
        stl_results: list[EpochResultSTL],
        weights: np.ndarray = None
    ) -> 'EpochResultMTL':
        '''
        Create an EpochResultMTL instance from a list of EpochResultSTL 
        instances. 

        Parameters
        ----------
        stl_results : list[EpochResultSTL]
            List of STL epoch results, one for each task.
        weights : np.ndarray, optional
            Weights for each task. If None, equal weights are assigned.
        
        Returns
        -------
        EpochResultMTL
            The corresponding MTL epoch result.
        '''
        # Default weights: equal weights if not provided
        if weights is None:
            weights = np.ones(len(stl_results)) / len(stl_results)
        
        # Aggregate
        losses = np.array([stl.loss for stl in stl_results])
        logits = [stl.logits for stl in stl_results]
        labels = [stl.labels for stl in stl_results]
        identifiers = [stl.identifiers for stl in stl_results]

        return cls(
            weights=weights,
            losses=losses,
            logits=logits,
            labels=labels,
            identifiers=identifiers
    )

    def to_stl(self) -> EpochResultSTL:
        '''
        Convert the MTL epoch result to an STL epoch result by computing a
        weighted loss and concatenating logits, labels, and identifiers across
        tasks.

        Returns
        -------
        EpochResultSTL
            The corresponding STL epoch result.
        '''
        # Compute weighted loss
        weighted_loss = np.sum(self.losses * self.weights)

        # Concatenate logits and labels across tasks
        logits = np.concatenate(self.logits, axis=0)
        labels = np.concatenate(self.labels, axis=0)

        # Flatten identifiers
        identifiers = [
            identifier
            for identifiers in self.identifiers 
            for identifier in identifiers
        ]

        return EpochResultSTL(
            loss=weighted_loss,
            logits=logits,
            labels=labels,
            identifiers=identifiers
        )

if __name__ == "__main__":

    # TEST EPOCHRESULTSTL
    epoch_result = EpochResultSTL(
        loss=0.5,
        logits=np.array([0.1, 0.2, 0.3]),
        labels=np.array([0, 1, 0]),
        identifiers=['id1', 'id2', 'id3']
    )

    print("Optimal threshold:", epoch_result.optimal_threshold)
    print("MTL result:", EpochResultMTL.from_stls(
        [epoch_result, epoch_result]
    ))

    # TEST EPOCHRESULTMTL
    epoch_result_mtl = EpochResultMTL(
        weights=np.array([0.5, 0.5]),
        losses=np.array([0.4, 0.6]),
        logits=[np.array([0.1, 0.2, 0.3]), np.array([0.3])],
        labels=[np.array([0, 1, 0]), np.array([1])],
        identifiers=[['id1', 'id2', 'id3'], ['id3']]
    )
    print(epoch_result_mtl.to_stl())