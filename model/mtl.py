"""
===============================================================================
Title:      MultiTask
Outline:    This module defines the MultiTask model, which consists of an
            encoder and multiple classification heads. The encoder can be frozen, 
            fine-tuned with LoRA, or fully trainable based on the specified 
            tuning strategy. The classification heads are always trainable. In
            between the encoder and the classification heads, a bottleneck
            layer is optionally applied to reduce the dimensionality of the 
            concatenated representations from the encoder.
Author:     Alejandro Sánchez Cano
Date:       11/10/2026
===============================================================================
"""

# Built-in modules
from typing import Literal

# Third-party modules
import peft
import torch
from torch import nn
from torchtyping import TensorType

# Custom modules
from model import utils

class MultiTask(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        bottleneck: nn.Module,
        heads: nn.ModuleList,
        tune: Literal['frozen', 'lora', 'full'] = 'frozen',
        peft_config: 'peft.LoraConfig' = None
    ):
        # Initialize nn.Module
        super().__init__()

        # Instance variables
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.heads = heads

        # Freeze the encoder parameters
        if tune == 'frozen':
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Apply LoRA
        elif tune == 'lora':
            if peft_config is None:
                raise ValueError("peft_config must be provided when tune is 'lora'")
            self.encoder = peft.get_peft_model(self.encoder, peft_config)

    def forward(
        self, 
        x: tuple[
            tuple[TensorType["batch", "seq_len"], TensorType["batch", "seq_len"]],
            tuple[TensorType["batch", "seq_len"], TensorType["batch", "seq_len"]]
        ],
        task_idx: int
    ) -> TensorType["batch"]:
        '''
        Forward step

        Parameters
        ----------
        x : tuple
            Tuple containing two tuples, each with (input_ids, attention_mask)
            for the two protein sequences.
        task_idx : int
            Determines which classification head to use for the forward pass.

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

        # Apply the bottleneck
        bottlenecked = self.bottleneck(pooled) # (batch, bottleneck_dim)

        # Forward pass through the classification head
        logits = self.heads[task_idx](bottlenecked) # (batch)

        return logits