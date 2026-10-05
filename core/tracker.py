"""
===============================================================================
Title:      Tracker
Outline:    Tracker class for monitoring and visualizing the training process
 of deep learning models.
Author:     Alejandro Sánchez Cano
Date:       29/09/2026
===============================================================================
"""

# Built-in modules
import copy

# Third-party modules
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from performance import Performance

class Tracker:

    def __init__(self):
        # Tracking history
        self.history = {
            'train': [],
            'val': []
        }
        self.performance = {
            'train': None,
            'val': None
        }
        # Best model tracking
        self.best_epoch = None
        self.best_model = None
        self.best_loss = float('inf')

    def track(
        self, 
        train: 'EpochResult',
        validation: 'EpochResult',
        model: 'torch.nn.Module'
    ) -> None:
        '''
        Track the train and validation results for the current epoch. Store
        the best model if the validation loss improves.

        Parameters
        ----------
        train : EpochResult
            Training results for the current epoch.
        validation : EpochResult
            Validation results for the current epoch.
        model : torch.nn.Module
            The model being trained.
        '''
        # Initialize performance object
        train_perf = Performance(
            true_labels=train.labels,
            predicted_logits=train.logits
        )
        val_perf = Performance(
            true_labels=validation.labels,
            predicted_logits=validation.logits
        )

        # Record results
        self.history['train'].append(train)
        self.history['val'].append(validation)
        self.performance['train'] = train_perf
        self.performance['val'] = val_perf

        # Update best model if validation loss improved
        if validation.loss < self.best_loss:
            self.best_loss = validation.loss
            self.best_epoch = len(self.history['val']) - 1
            self.best_model = copy.deepcopy(model.state_dict())

    def loss_curves(
        self,
        out_path: str | 'Path'
    ) -> None:
        '''
        Plot the loss curves for training and validation.

        Parameters
        ----------
        out_path : str or Path
            Path to save the loss curves plot.
        '''
        # Losses
        train_losses = [result.loss for result in self.history['train']]
        val_losses = [result.loss for result in self.history['val']]

        # Plotting
        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(out_path)

if __name__ == '__main__':
    # Epoch result class
    from dataclasses import dataclass
    @dataclass(frozen=True, slots=True)
    class EpochResult:
        loss: float
        logits: list
        labels: list
        identifiers: list

    # Mock model class
    import torch.nn as nn
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

    # Tracker
    tracker = Tracker()
    
    # Epoch loop
    for epoch in range(10):
        # Simulate epoch result
        train = EpochResult(
            loss=0.1 * (10 - epoch) if epoch != 6 else 0,
            logits=[epoch] * 5,
            labels=[epoch] * 5,
            identifiers=[f'id_{epoch}'] * 5
        )
        val = EpochResult(
            loss=0.15 * (10 - epoch) if epoch != 6 else 0,
            logits=[epoch] * 5,
            labels=[epoch] * 5,
            identifiers=[f'id_{epoch}'] * 5
        )
        # Track the result
        tracker.track(
            train=train, 
            validation=val,
            model=MockModel()
        )

    # Best epoch
    print(f"Best epoch based on validation loss: {tracker.best_epoch + 1}")

    # Plot loss curves
    tracker.loss_curves()