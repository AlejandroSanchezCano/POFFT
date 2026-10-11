"""
===============================================================================
Title:      Batch
Outline:    Batch class that conceptually represents a batch of data, but is 
            more convenient to work with than doing it manually. It is built
            taking into account the structure of the Dataset and the collate_fn
            elements.
Author:     Alejandro Sánchez Cano
Date:       11/10/2026
===============================================================================
"""

class Batch:

    def __init__(
        self,
        batch
    ):
        # Unpack the batch
        tokens1, tokens2 = batch['inputs']
        self.ids1, self.mask1 = tokens1
        self.ids2, self.mask2 = tokens2
        self.labels = batch['labels']
        self.identifiers = batch['identifiers']
    
    def __len__(self):
        return self.labels.size(0)

    def to(self, device: torch.device):
        '''
        Move the batch to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the batch to.
        '''
        self.ids1 = self.ids1.to(device, non_blocking=True)
        self.mask1 = self.mask1.to(device, non_blocking=True)
        self.ids2 = self.ids2.to(device, non_blocking=True)
        self.mask2 = self.mask2.to(device, non_blocking=True)
        self.labels = self.labels.to(device, non_blocking=True)

    @property
    def model_input(self) -> tuple[tuple, tuple]:
        '''
        Get the model input from the batch.

        Returns
        -------
        tuple
            The model input as a tuple of tuples.
        '''
        return (self.ids1, self.mask1), (self.ids2, self.mask2)