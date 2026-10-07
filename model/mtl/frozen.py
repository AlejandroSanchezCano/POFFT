
# Third-party modules
import torch
from torch import nn
from torchtyping import TensorType

class MultiTaskFrozen(nn.Module):

    def __init__(
        self, 
        encoder: nn.Module,
        bottleneck: nn.Module,
        heads: list[nn.Module],
    ):
        # Initialize nn.Module
        super().__init__()

        # Instance variables
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.heads = heads

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self, 
        x: tuple[
            tuple[TensorType["batch", "seq_len"], TensorType["batch", "seq_len"]],
            tuple[TensorType["batch", "seq_len"], TensorType["batch", "seq_len"]]
        ],
        task: str
    ) -> TensorType["batch"]:
        '''
        Forward step

        Parameters
        ----------
        x : tuple
            Tuple containing two tuples, each with (input_ids, attention_mask)
            for the two protein sequences.
        task : str
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
        logits = self.heads[task](bottlenecked) # (batch)

        return logits