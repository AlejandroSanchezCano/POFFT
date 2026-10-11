"""
===============================================================================
Title:      Result
Outline:    Result classes to store the results of a training or evaluation
            epoch.
Author:     Alejandro Sánchez Cano
Date:       11/10/2026
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

class Result(ABC):
    pass
    
@dataclass(frozen=True, slots=True)
class ResultSTL(Result):
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
class ResultMTL(Result):
    weights: np.ndarray
    losses: np.ndarray
    logits: list[np.ndarray]
    labels: list[np.ndarray]
    identifiers: list[list]

    @classmethod
    def from_stls(
        cls,
        stl_results: list[ResultSTL],
        weights: np.ndarray = None
    ) -> ResultMTL:
        '''
        Create an ResultMTL instance from a list of ResultSTL 
        instances. 

        Parameters
        ----------
        stl_results : list[ResultSTL]
            List of ResultSTL instances, one for each task.
        weights : np.ndarray, optional
            Weights for each task. If None, equal weights are assigned.
        
        Returns
        -------
        ResultMTL
            The corresponding MTL result.
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

    def to_stl(self) -> ResultSTL:
        '''
        Convert the MTL epoch result to an STL epoch result by computing a
        weighted loss and concatenating logits, labels, and identifiers across
        tasks.

        Returns
        -------
        ResultSTL
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

        return ResultSTL(
            loss=weighted_loss,
            logits=logits,
            labels=labels,
            identifiers=identifiers
        )

if __name__ == "__main__":

    # TEST RESULTSTL
    result = ResultSTL(
        loss=0.5,
        logits=np.array([0.1, 0.2, 0.3]),
        labels=np.array([0, 1, 0]),
        identifiers=['id1', 'id2', 'id3']
    )

    print("Optimal threshold:", result.optimal_threshold)
    print("MTL result:", ResultMTL.from_stls(
        [result, result]
    ))

    # TEST RESULTMTL
    result_mtl = ResultMTL(
        weights=np.array([0.5, 0.5]),
        losses=np.array([0.4, 0.6]),
        logits=[np.array([0.1, 0.2, 0.3]), np.array([0.3])],
        labels=[np.array([0, 1, 0]), np.array([1])],
        identifiers=[['id1', 'id2', 'id3'], ['id3']]
    )
    print(result_mtl.to_stl())