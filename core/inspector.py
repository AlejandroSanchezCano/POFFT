"""
===============================================================================
Title:      Model Inspector
Outline:    ModelInspector class for inspecting models and their properties:
            - Model parameters (trainable and non-trainable)
            - Number of parameters (trainable and non-trainable)
Author:     Alejandro Sánchez Cano
Date:       30/09/2026
===============================================================================
"""

# Built-in modules
from typing import Generator

class ModelInspector:
    def __init__(self, model):
        self.model = model

    def parameters(self, trainable: bool = False) -> Generator:
        """
        Get model parameters.

        Parameters
        ----------
        trainable : bool, optional
            If True, return only trainable parameters. If False, return all parameters.
            Default is False.

        Returns
        -------
        Generator
            Generator of model parameters.
        """
        if trainable:
            return (p for p in self.model.parameters() if p.requires_grad)
        else:
            return self.model.parameters()

    def num_parameters(self, trainable: bool = False) -> int:
        """
        Get the number of model parameters.

        Parameters
        ----------
        trainable : bool, optional
            If True, count only trainable parameters. If False, count all parameters.
            Default is False.

        Returns
        -------
        int
            Number of model parameters.
        """
        return sum(p.numel() for p in self.parameters(trainable=trainable))