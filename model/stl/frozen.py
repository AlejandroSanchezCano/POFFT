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

# Custom modules
from model import utils

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
            Logits.
        '''
        # Unpack inputs
        (ids1, mask1), (ids2, mask2) = x

        # Encode 
        encoded1 = utils.encode(encoder=self.encoder, inputs=(ids1, mask1)) # (batch, seq_len + 2, hidden_size)
        encoded2 = utils.encode(encoder=self.encoder, inputs=(ids2, mask2)) # (batch, seq_len + 2, hidden_size)

        # Pool representations
        pooled1 = utils.mean_pool(encoded1, mask1) # (batch, hidden_size)
        pooled2 = utils.mean_pool(encoded2, mask2) # (batch, hidden_size)

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
    from model.classification_head import ClassificationHead

    # Initialize model components
    esm2 = ESM2(config.ESM2_MODEL)
    classification_head = ClassificationHead(
        input_dim=esm2.hidden_size * 2,
        hidden_dims=config.CLASSIFICATION_HEAD_HIDDEN_DIMS,
    )
    model = Frozen(encoder=esm2.model, head=classification_head)

    # Inspect the model
    inspector = ModelInspector(model)
    print(f'Total parameters: {inspector.num_parameters(trainable=False)}')
    print(f'Trainable parameters: {inspector.num_parameters(trainable=True)}')

    # Test the model
    x = (
        (torch.randint(0, 20, (2, 10)), torch.ones((2, 10))), # (input_ids1, attention_mask1)
        (torch.randint(0, 20, (2, 15)), torch.ones((2, 15)))  # (input_ids2, attention_mask2)
    )
    logits = model(x)