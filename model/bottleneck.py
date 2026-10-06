"""
===============================================================================
Title:      Bottleneck
Outline:    Flexible bottleneck layer for reducing the dimensionality of the input
            before feeding it into a classification head.
Author:     Alejandro Sánchez Cano
Date:       06/10/2026
===============================================================================
"""

# Third-party modules
from torch import nn
from torchtyping import TensorType

class Bottleneck(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        bottleneck_dim: int, 
        dropout: float = 0.3
    ):
        # Initialize nn.Module
        super().__init__()

        # Instance variables
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout

        # Define the bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.input_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
    )

    def forward(
        self, 
        x: TensorType["batch_size", "input_dim"]
    ) -> TensorType["batch_size", "bottleneck_dim"]:
        '''
        Forward pass through the bottleneck

        Returns
        -------
        TensorType["batch_size", "bottleneck_dim"]
            The output of the bottleneck layer.
        '''
        return self.bottleneck(x)