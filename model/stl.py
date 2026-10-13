"""
===============================================================================
Title:      SingleTask
Outline:    This module defines the SingleTask model, which consists of an
            encoder and a classification head. The encoder can be frozen, 
            fine-tuned with LoRA, or fully trainable based on the specified 
            tuning strategy. The classification head is always trainable.
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

class SingleTask(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head: nn.Module,
        tune: Literal['frozen', 'lora', 'full'] = 'frozen',
        peft_config: 'peft.LoraConfig' = None
    ):
        # Initialize nn.Module
        super().__init__()

        # Instance variables
        self.encoder = encoder
        self.head = head

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
            tuple[TensorType["batch", "seq_len"], TensorType["batch", "seq_len"]],
            list[tuple[str, str]]
        ]
    ) -> TensorType["batch"]:
        '''
        Forward step

        Parameters
        ----------
        x : tuple
            Tuple containing two tuples, each with (input_ids, attention_mask)
            for the two protein sequences, and a list of identifier tuples.

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
    from peft import LoraConfig
    from core.inspector import ModelInspector
    from model.classification_head import ClassificationHead

    # Test different tuning strategies
    for tune in ['frozen', 'lora', 'full']:
        print(f'\nTesting SingleTask model with tune="{tune}"')

        # Initialize model components
        esm2 = ESM2(config.ESM2_MODEL)
        classification_head = ClassificationHead(
            input_dim=esm2.hidden_size * 2,
            hidden_dims=config.CLASSIFICATION_HEAD_HIDDEN_DIMS,
        )

        # Initialize the model
        model = SingleTask(
            encoder=esm2.model,
            head=classification_head,
            tune=tune,
            peft_config=LoraConfig(**config.LORA_CONFIG) if tune == 'lora' else None
        )

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