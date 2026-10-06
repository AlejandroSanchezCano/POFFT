"""
===============================================================================
Title:      FullFineTune
Outline:    This module defines the FullFineTune model, which consists of an
            encoder (ESM2) and classification head, both trainable, allowing
            for full fine-tuning of the model.
Author:     Alejandro Sánchez Cano
Date:       05/10/2026
===============================================================================
"""

# Third-party modules
import torch
from torch import nn
from torchtyping import TensorType

class FullFineTune(nn.Module):

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

if __name__ == "__main__":
    # Test the model
    from misc import config
    from tool.esm2 import ESM2
    from core.inspector import ModelInspector
    from classification_head import ClassificationHead

    # Initialize model components
    esm2 = ESM2(config.ESM2_MODEL)
    classification_head = ClassificationHead(
        input_dim=esm2.hidden_size * 2,
        hidden_dims=config.CLASSIFICATION_HEAD_HIDDEN_DIMS,
    )
    model = FullFineTune(encoder=esm2.model, head=classification_head)

    # Inspect the model
    inspector = ModelInspector(model)
    print(f'Total parameters: {inspector.num_parameters(trainable=False)}')
    print(f'Trainable parameters: {inspector.num_parameters(trainable=True)}')