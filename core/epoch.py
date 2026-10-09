"""
===============================================================================
Title:      Epoch
Outline:    Epoch classes to handle training and evaluation of a deep learning
            model for one epoch. It supports Automatic Mixed Precision (AMP)
            for faster training and reduced memory usage. 
Author:     Alejandro Sánchez Cano
Date:       08/10/2026
===============================================================================
"""

# Built-in modules
from abc import ABC, abstractmethod

# Third-party modules
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

# Custom modules
from misc import config
from core.loader_cycler import LoaderCycler
from core.task_weight import WeightingStrategy
from core.result import EpochResult, EpochResultSTL, EpochResultMTL

###############################################################################
#######                      ABSTRACT BASE CLASS                        #######
###############################################################################

class Epoch(ABC):

    @property
    def model(self) -> nn.Module:
        '''Get the model.'''
        return self._model
    
    @model.setter
    def model(self, model: nn.Module) -> None:
        '''Set the model and move it to the specified device.'''
        self._model = model.to(self.device)

    @abstractmethod
    def train(self, *args, **kwargs) -> EpochResult:
        '''Train the model for one epoch.'''
        pass

    @torch.inference_mode()
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> EpochResult:
        '''Evaluate the model for one epoch.'''
        pass

    @abstractmethod
    def _run(self, *args, **kwargs) -> EpochResult:
        '''Run one epoch of training or evaluation.'''
        pass

###############################################################################
#######                      SINGLE TASK LEARNING                       #######
###############################################################################

class EpochSTL:

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

    def train(
        self, 
        dataloader: DataLoader,
    ) -> EpochResultSTL:
        '''Train the model for one epoch.'''
        return self._run(dataloader, training=True)

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> EpochResultSTL:
        '''Evaluate the model for one epoch.'''
        return self._run(dataloader, training=False)

    def _run(
        self, 
        dataloader: DataLoader,
        training: bool,
    ) -> EpochResultSTL:
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
        EpochResultSTL
            EpochResult object containing loss, logits, labels, and identifiers.
        '''
        # Set mode -> dropout, batchnorm, etc.
        self.model.train(training)

        # Initialize 
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

        return EpochResultSTL(
            loss=total_loss / total_samples,
            logits=total_logits,
            labels=total_labels,
            identifiers=total_identifiers
        )

###############################################################################
#######                     MULTIPLE TASK LEARNING                      #######
###############################################################################

class EpochMTL(Epoch):

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        weighting_strategy: WeightingStrategy,
        device: torch.device = torch.device('cuda'),
        enable_amp: bool = True,
    ):
        # Instance variables
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.weighting = weighting_strategy

        # Automatic Mixed Precision (AMP)
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.enable_amp
        )

    def train(
        self, 
        cycler: LoaderCycler,
    ) -> EpochResultMTL:
        '''Train the model for one epoch.'''
        return self._run(cycler, training=True)

    @torch.inference_mode()
    def evaluate(
        self,
        cycler: LoaderCycler,
    ) -> EpochResultMTL:
        '''Evaluate the model for one epoch.'''
        return self._run(cycler, training=False)

    def _run(
        self, 
        cycler: LoaderCycler,
        training: bool,
    ) -> EpochResultMTL:
        '''
        Run one epoch of training or evaluation.

        Parameters
        ----------
        cycler : LoaderCycler
            LoaderCycler for the dataset.
        training : bool
            If True, run in training mode; otherwise, run in evaluation mode.
        
        Returns
        -------
        EpochResultMTL
            EpochResult object containing loss, logits, labels, and identifiers.
        '''
        # Set mode -> dropout, batchnorm, etc.
        self.model.train(training)

        # Initialize
        num_tasks = len(cycler)
        total_samples = np.zeros(num_tasks, dtype=np.int64)
        total_losses = np.zeros(num_tasks, dtype=np.float64)
        total_logits = [[] for _ in range(num_tasks)]
        total_labels = [[] for _ in range(num_tasks)]
        total_identifiers = [[] for _ in range(num_tasks)]

        # Iterate over loader cycler
        desc = "Training" if training else "Evaluating"
        for batches in tqdm(cycler, desc=desc, unit="batch"):

            # Initialize batch values
            weighted_losses = np.zeros(len(batches))

            # Iterate over tasks
            for task_idx, batch in enumerate(batches):

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
                task_labels = labels.to(self.device, non_blocking=True)

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
                    task_logits = self.model(model_input, task_idx=task_idx)
                    task_loss = self.loss_fn(task_logits, task_labels)    
                
                # Weight task loss
                # FIXME: item() destroys the computational graph
                weighted_loss = self.weighting.weights[task_idx] * task_loss
                weighted_losses[task_idx] = weighted_loss.item()

                # Accumulate
                task_samples = task_labels.size(0)
                total_samples[task_idx] += task_samples
                total_losses[task_idx] += weighted_loss.item() * task_samples
                total_logits[task_idx].append(task_logits.detach().cpu())
                total_labels[task_idx].append(task_labels.detach().cpu())
                total_identifiers[task_idx].extend(identifiers)

            # Sum weighted losses
            loss = np.sum(weighted_losses)

            # Backward pass and optimization
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        # Convert to numpy arrays
        for idx in range(len(cycler)):
            #FIXME: consider that some tasks may not be present!
            total_logits[idx] = torch.cat(total_logits[idx]).numpy()
            total_labels[idx] = torch.cat(total_labels[idx]).numpy()

        return EpochResultMTL(
            weights=self.weighting.weights,
            #FIXME: consider that some tasks may not be present! Use
            # np.divide(total_losses, total_samples, 
            # out=np.zeros_like(total_losses), where=total_samples > 0)
            losses=total_losses / total_samples,
            logits=total_logits,
            labels=total_labels,
            identifiers=total_identifiers
        )

                  
