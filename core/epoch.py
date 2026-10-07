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
    logits: 'np.ndarray'
    labels: 'np.ndarray'
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
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
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
        desc = "Training" if training else "Evaluating"
        for batch in tqdm(dataloader, desc=desc, unit="batch"):

            # Unpack batch
            tokens1, tokens2 = batch['inputs']
            ids1, mask1 = tokens1
            ids2, mask2 = tokens2
            labels = batch['labels']
            identifiers = batch['identifiers']

            # Move to device
            ids1 = ids1.to(self.device, non_blocking=True)
            mask1 = mask1.to(self.device, non_blocking=True)
            ids2 = ids2.to(self.device, non_blocking=True)
            mask2 = mask2.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Construct model input
            model_input = ((ids1, mask1), (ids2, mask2))

            # Zero gradients
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            with torch.amp.autocast(
                device_type=self.device.type, 
                enabled=self.enable_amp
            ):
                logits = self.model(model_input)
                loss = self.loss_fn(logits, labels)

            # Backward pass and optimization
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Accumulate
            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            total_logits.append(logits.detach().cpu())
            total_labels.append(labels.detach().cpu())
            total_identifiers.extend(identifiers)

        # Convert to numpy arrays
        total_logits = torch.cat(total_logits).numpy()
        total_labels = torch.cat(total_labels).numpy()

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