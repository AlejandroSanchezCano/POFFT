
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
        x
    ) -> TensorType["batch"]:

        # Frozen encoder forward pass
        p1input_ids, p1attention_mask = x[0]
        p2input_ids, p2attention_mask = x[1]

        with torch.no_grad():
            p1outputs = self.encoder(
                input_ids=p1input_ids,
                attention_mask=p1attention_mask,
            )

            p2outputs = self.encoder(
                input_ids=p2input_ids,
                attention_mask=p2attention_mask,
            )

        # Pool embeddings
        p1hidden = p1outputs.last_hidden_state[:, 1:-1, :] # (batch, seq_len, hidden_size)
        p2hidden = p2outputs.last_hidden_state[:, 1:-1, :] # (batch, seq_len, hidden_size)
        print(f'p1hidden shape: {p1hidden.shape}, p2hidden shape: {p2hidden.shape}')
        hidden = torch.cat([p1hidden, p2hidden], dim=2) # (batch, seq_len, hidden_size*2)
        print(f'Concatenated hidden shape: {hidden.shape}')
        pooled = hidden.mean(dim=1) # (batch, hidden_size)
        print(f'Pooled shape: {pooled.shape}')

        # Pass through the classification head
        logits = self.head(pooled) # (batch)
        print(f'Logits shape: {logits.shape}')

        return logits