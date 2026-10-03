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
        self.best_model = None

    def __call__(
        self,
        loss: float,
        model: 'torch.nn.Module',
    ) -> bool:
        '''
        Check if the training should stop.

        Parameters
        ----------
        loss : float
            Validation/test loss.
        model : torch.nn.Module
            Model to save if the loss improves.

        Returns
        -------
        bool
            True if the training should stop, False otherwise.
        '''
        # Check if the loss has improved
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())
            return False
        else:
            # Decrease patience counter
            self.patience -= 1
            # Check if patience has run out
            if self.patience <= 0:
                return True
            return False

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Dummy model
    model = nn.Linear(10, 1)

    # Early stopping
    early_stop = EarlyStop(patience=3, min_delta=0.01)

    # Simulate training loop
    losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41]
    for epoch, loss in enumerate(losses):
        print(f"Epoch {epoch+1}, Loss: {loss}, Patience: {early_stop.patience}")
        if early_stop(loss, model):
            print("Early stopping triggered.")
            break