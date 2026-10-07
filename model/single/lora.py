"""
===============================================================================
Title:      LoRAFineTune
Outline:    This module defines the LoRAFineTune model, which consists of an
            encoder (ESM2) and classification head, both trainable, allowing
            for fine-tuning of the model using LoRA (Low-Rank Adaptation).
Author:     Alejandro Sánchez Cano
Date:       05/10/2026
===============================================================================
"""

# Third-party modules
import peft
import torch
from torch import nn
from torchtyping import TensorType

# Custom modules
from model import utils

class LoRAFineTune(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        head: nn.Module,
        lora_config: 'LoraConfig'
    ):
        # Initialize nn.Module
        super().__init__()

        # Instance variables
        self.encoder = peft.get_peft_model(encoder, lora_config)
        self.head = head

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
    from peft import LoraConfig
    from core.inspector import ModelInspector
    from model.classification_head import ClassificationHead

    # Initialize model components
    esm2 = ESM2(config.ESM2_MODEL)
    classification_head = ClassificationHead(
        input_dim=esm2.hidden_size * 2,
        hidden_dims=config.CLASSIFICATION_HEAD_HIDDEN_DIMS,
    )
    model = LoRAFineTune(
        encoder=esm2.model, 
        head=classification_head,
        lora_config=LoraConfig(**config.LORA_CONFIG)
    )
    model.encoder.print_trainable_parameters() 

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