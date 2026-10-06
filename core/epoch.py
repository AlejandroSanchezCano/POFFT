"""
===============================================================================
Title:      Tracker
Outline:    Tracker class for monitoring and visualizing the training process 
            of deep learning models.
Author:     Alejandro Sánchez Cano
Date:       05/10/2026
===============================================================================
"""

# Built-in modules
from dataclasses import dataclass

# Third-party modules
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

@dataclass(frozen=True, slots=True)
class EpochResult:
    loss: float
    logits: list | 'np.ndarray'
    labels: list | 'np.ndarray'
    identifiers: list

    def to_dataframe(self) -> pd.DataFrame:
        '''Convert the epoch result to a pandas DataFrame.'''
        return pd.DataFrame({
            'logits': self.logits,
            'labels': self.labels,
            'identifiers': self.identifiers
        })

class Epoch:

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device('cuda'),
        enable_amp: bool = True,
    ):
        # Instance variables
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.enable_amp = enable_amp

        # Automatic Mixed Precision (AMP)
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.enable_amp
        )

    @property
    def model(self) -> nn.Module:
        '''Get the model.'''
        return self._model
    
    @model.setter
    def model(self, model: nn.Module) -> None:
        '''Set the model and move it to the specified device.'''
        self._model = model.to(self.device)

    def train(
        self, 
        dataloader: DataLoader,
    ) -> EpochResult:
        '''Train the model for one epoch.'''
        return self._run(dataloader, training=True)

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> EpochResult:
        '''Evaluate the model for one epoch.'''
        return self._run(dataloader, training=False)

    def _run(
        self, 
        dataloader: DataLoader,
        training: bool,
    ) -> EpochResult:
        '''
        Run one epoch of training or evaluation.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for the dataset.
        training : bool
            If True, run in training mode; otherwise, run in evaluation mode.
        
        Returns
        -------
        EpochResult
            EpochResult object containing loss, logits, labels, and identifiers.
        '''
        # Set mode -> dropout, batchnorm, etc.
        self.model.train(training)

        # Initialize metrics
        total_loss = 0.0
        total_samples = 0
        total_logits = []
        total_labels = []
        total_identifiers = []

        # Iterate over batches
        for batch in tqdm(dataloader, desc="Batches", unit="batch"):

            # Unpack batch
            tokens1, tokens2 = batch['inputs']
            ids1, mask1 = tokens1
            ids2, mask2 = tokens2
            labels = batch['labels']
            identifiers = batch['identifiers']

            # Move to device
            ids1 = ids1.to(self.device)
            mask1 = mask1.to(self.device)
            ids2 = ids2.to(self.device)
            mask2 = mask2.to(self.device)
            labels = labels.to(self.device)

            # Construct model input
            model_input = ((ids1, mask1), (ids2, mask2))

            # Zero gradients
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.enable_amp):
                logits = self.model(
                    model_input,
                    identifiers
                )
                loss = self.loss_fn(logits, labels)

            # Backward pass and optimization
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Accumulate
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            total_logits.extend(logits.detach().cpu().numpy())
            total_labels.extend(labels.detach().cpu().numpy())
            total_identifiers.extend(identifiers)

        return EpochResult(
            loss=total_loss / total_samples,
            logits=total_logits,
            labels=total_labels,
            identifiers=total_identifiers
        )

if __name__ == "__main__":
    import numpy as np
    epoch_result = EpochResult(
        loss=0.5,
        logits=np.array([0.1, 0.2, 0.3]),
        labels=np.array([0, 1, 0]),
        identifiers=['id1', 'id2', 'id3']
    )
    print(epoch_result.to_dataframe())