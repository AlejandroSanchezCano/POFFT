
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
        ],
        identifiers: list[tuple[str, str]],
        *args,
        **kwargs
    ) -> TensorType["batch"]:
        '''
        Forward step

        Parameters
        ----------
        x : tuple
            Tuple containing two tuples, each with (input_ids, attention_mask)
            for the two protein sequences.
        identifiers : list[tuple[str, str]]
            List of tuples containing the identifiers for the two protein 
            sequences in each pair.

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

        # Apply the bottleneck
        bottlenecked = self.bottleneck(pooled) # (batch, bottleneck_dim)

        #TODO: add a mechanism to select the appropriate head based on the identifiers or some other criteria. For now, we will assume a single head for simplicity.


        # Forward pass through the classification head
        logits = self.head(bottlenecked) # (batch)

        return logits
