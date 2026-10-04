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

    def _encode(
        self, 
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"]
    ) -> TensorType["batch", "hidden_size"]:
        '''
        Combine encoder and mean pooling to encode the input sequences into
        fixed-size embeddings.

        Parameters
        ----------
        input_ids : TensorType["batch", "seq_len"]
            Input token IDs for the sequences.
        
        attention_mask : TensorType["batch", "seq_len"]
            Attention mask for the sequences.

        Returns
        -------
        TensorType["batch", "hidden_size"]
            Encoded representations of the input sequences.
        '''
        # Forward pass through the encoder
        with torch.no_grad():
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Obtain embeddings (mean pooling with attention mask)
        hidden = output.last_hidden_state[:, 1:-1, :] # (batch, seq_len, hidden_size)
        mask = attention_mask[:, 1:-1].unsqueeze(-1) # (batch, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1) # (batch, hidden_size)

        return pooled

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

        # Encode the input sequences
        pooled1 = self._encode(ids1, mask1)
        pooled2 = self._encode(ids2, mask2)

        # Concatenate the pooled embeddings
        pooled = torch.cat([pooled1, pooled2], dim=-1) # (batch, hidden_size*2)

        # Forward pass through the classification head
        logits = self.head(pooled) # (batch)

        return logits