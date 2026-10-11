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
from core.batch import Batch
from core.loader_cycler import LoaderCycler
from core.task_weight import WeightingStrategy
from core.result import Result, ResultSTL, ResultMTL

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
    def train(self, *args, **kwargs) -> Result:
        '''Train the model for one epoch.'''
        pass

    @torch.inference_mode()
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Result:
        '''Evaluate the model for one epoch.'''
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
    ) -> ResultSTL:
        '''Train the model for one epoch.'''
        return self._run(dataloader, training=True)

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> ResultSTL:
        '''Evaluate the model for one epoch.'''
        return self._run(dataloader, training=False)

    def _run(
        self, 
        dataloader: DataLoader,
        training: bool,
    ) -> ResultSTL:
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
        ResultSTL
            Result object containing loss, logits, labels, and identifiers.
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

            # Batch shenanigans
            batch = Batch(batch)
            batch.to(self.device)
            model_input = batch.model_input

            # Zero gradients
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            with torch.amp.autocast(
                device_type=self.device.type, 
                enabled=self.enable_amp
            ):
                logits = self.model(model_input)
                loss = self.loss_fn(logits, batch.labels)

            # Backward pass and optimization
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Accumulate
            batch_size = len(batch)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            total_logits.append(logits.detach().cpu())
            total_labels.append(batch.labels.detach().cpu())
            total_identifiers.extend(batch.identifiers)

        # Convert to numpy arrays
        total_logits = torch.cat(total_logits).numpy()
        total_labels = torch.cat(total_labels).numpy()

        return ResultSTL(
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
        weighting: WeightingStrategy,
        device: torch.device = torch.device('cuda'),
        enable_amp: bool = True,
    ):
        # Instance variables
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.enable_amp = enable_amp
        self.weighting = weighting

        # Automatic Mixed Precision (AMP)
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.enable_amp
        )

    def train(
        self, 
        cycler: LoaderCycler,
    ) -> ResultMTL:
        '''
        Run one epoch of training.

        Parameters
        ----------
        cycler : LoaderCycler
            LoaderCycler for the dataset.
        
        Returns
        -------
        ResultMTL
            Result object containing loss, logits, labels, and identifiers.
        '''
        # Set mode -> dropout, batchnorm, etc.
        self.model.train()

        # Initialize
        num_tasks = len(cycler.loaders)
        samples_per_task = np.zeros(num_tasks, dtype=np.int64)
        total_loss = 0.0
        losses_per_task = np.zeros(num_tasks, dtype=np.float64)
        logits_per_task = [[] for _ in range(num_tasks)]
        labels_per_task = [[] for _ in range(num_tasks)]
        identifiers_per_task = [[] for _ in range(num_tasks)]

        # Iterate over loader cycler
        for batches in tqdm(cycler, desc='Training', unit="batch"):
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Initialize batch values
            weighted_losses = []

            # Iterate over tasks
            for task_idx, batch in enumerate(batches):

                # Batch shenanigans
                batch = Batch(batch)
                batch.to(self.device)
                model_input = batch.model_input
                
                # Forward pass with AMP
                with torch.amp.autocast(
                    device_type=self.device.type, 
                    enabled=self.enable_amp
                ):
                    task_logits = self.model(model_input, task_idx=task_idx)
                    task_loss = self.loss_fn(task_logits, batch.labels)    
                
                # Weight task loss
                weighted_loss = self.weighting.weights[task_idx] * task_loss
                weighted_losses.append(weighted_loss)

                # Accumulate
                task_samples = len(batch)
                samples_per_task[task_idx] += task_samples
                losses_per_task[task_idx] += task_loss.item() * task_samples
                logits_per_task[task_idx].append(task_logits.detach().cpu())
                labels_per_task[task_idx].append(batch.labels.detach().cpu())
                identifiers_per_task[task_idx].extend(batch.identifiers)

            # Sum weighted losses
            loss = torch.stack(weighted_losses).sum()

            # Backward pass and optimization
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate total loss
            samples = len(batch) * num_tasks
            total_loss += loss.detach().item() * samples

        # Convert to numpy arrays
        for idx in range(num_tasks):
            logits_per_task[idx] = torch.cat(logits_per_task[idx]).numpy()
            labels_per_task[idx] = torch.cat(labels_per_task[idx]).numpy()

        return ResultMTL(
            #loss=total_loss / sum(samples_per_task),
            weights=self.weighting.weights,
            losses=losses_per_task / samples_per_task,
            logits=logits_per_task,
            labels=labels_per_task,
            identifiers=identifiers_per_task
        )

# Here there have been two losses computed: the per-task losses and the
# aggregated loss. The aggregated loss is used for backpropagation, while the
# per-task losses are used for evaluation and tracking. However, they are
# equivalent and one can be derived from the other. In fact, having an
# ResultMTL object called er, one can derive that:
# er.loss == sum(er.losses * er.weights)
# This is true regardless of the number of samples per tasks, as the losses are
# already normalized by the number of samples per task. 
# Ultimately, the aggregated loss is not reported in the ResultMTL object, 
# as it can be easily derived from the per-task losses and weights. This design
# choice brings ResultMTL closer to the STL ResultSTL, which only 
# reports the loss for the single task.

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader,
        task_idx: int
    ) -> ResultSTL:
        '''
        Run one epoch of evaluation.
        Since we evaluate one task at a time, we can reuse 99% of the code from
        the STL epoch class. The only difference is that we need to pass the
        task index to the model.

        Parameters
        ---------- 
        dataloader : DataLoader
            DataLoader for the dataset.
        task_idx : int
            Index of the task to evaluate.
        
        Returns
        -------
        ResultSTL
            Result object containing loss, logits, labels, and identifiers.
        '''
        # Set mode -> dropout, batchnorm, etc.
        self.model.eval()

        # Initialize 
        total_loss = 0.0
        total_samples = 0
        total_logits = []
        total_labels = []
        total_identifiers = []

        # Iterate over batches
        desc = f"Evaluating Task {task_idx}"
        for batch in tqdm(dataloader, desc=desc, unit="batch"):

            # Batch shenanigans
            batch = Batch(batch)
            batch.to(self.device)
            model_input = batch.model_input
            
            # Forward pass with AMP
            with torch.amp.autocast(
                device_type=self.device.type, 
                enabled=self.enable_amp
            ):
                logits = self.model(model_input, task_idx=task_idx)
                loss = self.loss_fn(logits, batch.labels)

            # Accumulate
            batch_size = len(batch)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
            total_logits.append(logits.detach().cpu())
            total_labels.append(batch.labels.detach().cpu())
            total_identifiers.extend(batch.identifiers)

        # Convert to numpy arrays
        total_logits = torch.cat(total_logits).numpy()
        total_labels = torch.cat(total_labels).numpy()

        return ResultSTL(
            loss=total_loss / total_samples,
            logits=total_logits,
            labels=total_labels,
            identifiers=total_identifiers
        )        