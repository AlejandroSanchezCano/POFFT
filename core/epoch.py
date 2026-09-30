"""
===============================================================================
Title:      Epoch
Outline:    Epoch class for training and evaluating deep learning models.
Author:     Alejandro Sánchez Cano
Date:       29/09/2026
===============================================================================
"""

# Built-in modules
from dataclasses import dataclass

# Third-party modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

@dataclass(frozen=True, slots=True)
class EpochResult:
    loss: float
    logits: list
    labels: list
    identifiers: list

class Epoch:

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device | str = 'cuda',
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

        # Move model to device
        self.model.to(self.device)

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
        for batch in dataloader:

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

            # Zero gradients
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.enable_amp):
                logits = self.model((ids1, mask1), (ids2, mask2))
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
