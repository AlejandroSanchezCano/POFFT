"""
===============================================================================
Title:      Classification Head
Outline:    Flexible classification head for binary classification tasks that
            can be easily integrated into any model.
Author:     Alejandro Sánchez Cano
Date:       24/09/2026
===============================================================================
"""

# Third-party modules
from torch import nn
from torchtyping import TensorType

class ClassificationHead(nn.Module):
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3
    ):
        # Initialize nn.Module
        super().__init__()

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Build the layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            layers.extend(layer)
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))

        # Create the sequential model
        self.ffn = nn.Sequential(*layers)

    def forward(
        self, 
        x: TensorType['batch', 'input_dim']
    ) -> TensorType['batch']:
        '''
        Forward pass through the classification head to obtain logits. 
        Since the final layer is a single neuron for binary classification, the
        output is squeezed to remove the last dimension.

        Parameters
        ----------
        x : TensorType['batch', 'input_dim']
            Input tensor

        Returns
        -------
        TensorType['batch']
            Logits.
        '''
        return self.ffn(x).squeeze(-1)

if __name__ == "__main__":
    import torch
    head = ClassificationHead(input_dim=1280)
    x = torch.randn(10, 1280)
    output = head(x)
    print(output.shape)  #torch.Size([10])
    