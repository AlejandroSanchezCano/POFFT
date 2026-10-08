"""
===============================================================================
Title:      Utils
Outline:    Utility functions for the models:
            - encode: Encodes input sequences using a given encoder.
            - mean_pool: Applies mean pooling to the encoded representations.
Author:     Alejandro Sánchez Cano
Date:       07/10/2026
===============================================================================
"""

# Built-in modules
import contextlib
from typing import Any

# Third-party modules
import torch
from torch import nn
from torchtyping import TensorType

def mean_pool(
    encoded: TensorType["batch", "seq_len", "hidden_size"],
    attention_mask: TensorType["batch", "seq_len"]
) -> TensorType["batch", "hidden_size"]:
    '''
    Apply mean pooling to the encoded representations with attention mask and
    disregarding the special tokens.

    Parameters
    ----------
    encoded : TensorType["batch", "seq_len", "hidden_size"]
        Encoded representations of the input sequences.

    attention_mask : TensorType["batch", "seq_len"]
        Attention mask for the sequences.

    Returns
    -------
    TensorType["batch", "hidden_size"]
        Pooled representations of the input sequences.
    '''
    encoded = encoded[:, 1:-1, :] # (batch, seq_len, hidden_size)
    mask = attention_mask[:, 1:-1].unsqueeze(-1) # (batch, seq_len, 1)
    pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1) # (batch, hidden_size)

    return pooled

def encode(
    encoder: nn.Module,
    inputs: tuple
) -> TensorType["batch", "seq_len", "hidden_size"]:
    '''
    Encode the input sequences using the given encoder.

    Parameters
    ----------
    encoder : nn.Module
        The encoder model to use for encoding the input sequences.
    
    inputs : tuple
        Model inputs.
    
    Returns
    -------
    TensorType["batch", "seq_len", "hidden_size"]
        Encoded representations of the input sequences.
    '''
    # Choose context manager
    is_frozen = not any(param.requires_grad for param in encoder.parameters())
    grad_context = torch.no_grad if is_frozen else contextlib.nullcontext()

    # Forward pass through the encoder
    with grad_context():
        output = encoder(*inputs)

    return output.last_hidden_state