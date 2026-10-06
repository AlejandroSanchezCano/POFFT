"""
===============================================================================
Title:      Early stopping
Outline:    EarlyStopping class to halt training when the validation/test loss
            does not improve after a certain number of epochs.
Author:     Alejandro Sánchez Cano
Date:       29/09/2026
===============================================================================
"""

# Built-in modules
import copy

class EarlyStop:

    def __init__(
        self,
        patience: int = 4, 
        min_delta: float = 0.0001
    ):
        '''
        Class constructor.

        Parameters
        ----------
        patience : int, optional
            Number of epochs with no improvement after which training will be 
            stopped, by default 4.
        
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as an 
            improvement, by default 0.0001
        '''
        # Validate
        if patience < 1:
            raise ValueError("Patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("Min delta must be non-negative.")
        
        # Instance variables
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(
        self,
        loss: float,
    ) -> bool:
        '''
        Check if the training should stop.

        Parameters
        ----------
        loss : float
            Validation/test loss.

        Returns
        -------
        bool
            True if the training should stop, False otherwise.
        '''
        # Check if the loss has improved
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False

        # Increase patience counter
        self.counter += 1
        # Check if patience has run out
        return self.counter >= self.patience

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Dummy model
    model = nn.Linear(10, 1)

    # Early stopping
    early_stop = EarlyStop(patience=4, min_delta=0)

    # Simulate training loop
    #losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41]
    losses = [293, 294, 300, 302, 291, 297, 300, 305, 310, 315, 320]
    for epoch, loss in enumerate(losses):
        print(f"Epoch {epoch+1}, Loss: {loss}, Patience: {early_stop.counter}")
        if early_stop(loss):
            print("Early stopping triggered.")
            break