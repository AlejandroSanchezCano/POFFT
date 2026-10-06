"""
===============================================================================
Title:      Loss Functions
Outline:    Custom loss functions for training models.
            - Binary Focal Loss
Author:     Alejandro Sánchez Cano
Date:       06/10/2026
===============================================================================
"""

# Third-party modules
import torch
import torch.nn as nn
from torchtyping import TensorType

class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float | int = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean"
    ):
        # Initialize nn.Module
        super().__init__()
        # Validate arguments
        if gamma < 0.0:
            raise ValueError("Gamma must be non-negative.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be in the range [0, 1].")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("Reduction must be one of 'mean', 'sum', or 'none'.")
        # Instance variables
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: TensorType['batch'],
        targets: TensorType['batch'],
    ) -> torch.Tensor:
        '''
        Compute the binary focal loss.

        Parameters
        ----------
        logits : TensorType['batch']
            Model logits (raw outputs before sigmoid).
        targets : TensorType['batch']
            Binary targets (0 or 1), same shape as logits.

        Returns
        -------
        torch.Tensor
            Computed focal loss, reduced according to self.reduction.
        '''
        # Same dtype
        targets = targets.to(dtype=logits.dtype)
        # BCE loss from logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )
        # Compute probabilities p_t
        p_t = torch.exp(-bce_loss)
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Compute focal loss
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss

if __name__ == "__main__":
    # Test losses
    logits = torch.tensor([0.0, 1.0, -1.0, 2.0])
    targets = torch.tensor([0, 1, 0, 1])
    loss = BinaryFocalLoss()(logits, targets)
    print(f"Focal Loss: {loss.item()}")
