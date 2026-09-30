"""
===============================================================================
Title:      Frozen
Outline:    This module defines the Frozen model, which consists of a frozen
            encoder (ESM2) and a trainable classification head. The encoder's
            parameters are frozen to prevent updates during training, while the
            classification head is trainable.
Author:     Alejandro Sánchez Cano
Date:       24/09/2026
===============================================================================
"""

# Third-party modules
import torch
from torch import nn
from torchtyping import TensorType

class Frozen(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head: nn.Module
    ):
        # Initialize nn.Module
        super().__init__()

        # Instance variables
        self.encoder = encoder
        self.head = head

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self, 
        x: tuple[
            tuple[TensorType["batch", "seq_len"], TensorType["batch", "seq_len"]],
            tuple[TensorType["batch", "seq_len"], TensorType["batch", "seq_len"]]
        ]
    ) -> TensorType["batch"]:
        '''
        Forward step

        Parameters
        ----------
        x : tuple
            Tuple containing two tuples, each with (input_ids, attention_mask)
            for the two protein sequences.

        Returns
        -------
        TensorType["batch"]
            Logits for the classification task.
        '''
        # Unpack inputs
        (ids1, mask1), (ids2, mask2) = x

        # Forward pass through the encoder
        with torch.no_grad():
            output1 = self.encoder(
                input_ids=ids1,
                attention_mask=mask1,
            )
            output2 = self.encoder(
                input_ids=ids2,
                attention_mask=mask2,
            )

        # Obtain embeddings
        hidden1 = output1.last_hidden_state[:, 1:-1, :] # (batch, seq_len, hidden_size)
        hidden2 = output2.last_hidden_state[:, 1:-1, :] # (batch, seq_len, hidden_size)
        
        # Pool embeddings (mean pooling with attention mask)
        mask1 = mask1[:, 1:-1].unsqueeze(-1) # (batch, seq_len, 1)
        mask2 = mask2[:, 1:-1].unsqueeze(-1) # (batch, seq_len, 1)
        pooled1 = (hidden1 * mask1).sum(dim=1) / mask1.sum(dim=1) # (batch, hidden_size)
        pooled2 = (hidden2 * mask2).sum(dim=1) / mask2.sum(dim=1) # (batch, hidden_size)
        pooled = torch.cat([pooled1, pooled2], dim=1) # (batch, hidden_size*2)

        # Forward pass through the classification head
        logits = self.head(pooled) # (batch)

        return logits