"""
===============================================================================
Title:      ESM2
Outline:    ESM2 class to interact with HuggingFace's ESM2 model checkpoints. 
            It supports:
            - Loading different ESM2 models
            - Validating and tokenizing protein sequences
            - Running the model for specific layers
            - Extracting representations (per-residue and per-sequence)

            It does not support:
            - Estimating perplexity (fast and pseudo)
            - Plotting attention maps
            - Getting contact maps
Docs:       https://huggingface.co/facebook/models?search=esm
Author:     Alejandro Sánchez Cano
Date:       22/09/2026
===============================================================================
"""

# Built-in modules
import re
from typing import Literal

# Third-party modules
import torch
import transformers

class ESM2:

    model2checkpoint = {
        '8M' : 'facebook/esm2_t6_8M_UR50D',
        '35M' : 'facebook/esm2_t12_35M_UR50D',
        '150M' : 'facebook/esm2_t30_150M_UR50D',
        '650M' : 'facebook/esm2_t33_650M_UR50D',
        '3B' : 'facebook/esm2_t36_3B_UR50D',
        '15B' : 'facebook/esm2_t48_15B_UR50D'
    }

    def __init__(self, model: str):
        self.model, self.tokenizer = self._load(model)
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.num_params = sum(
            param.numel() 
            for param in self.model.parameters()
        )
        self.output = None

    def _load(self, model: str) -> tuple['EsmModel', 'EsmTokenizer']:
        '''
        Loads the ESM2 model and tokenizer from HuggingFace Transformers.

        Parameters
        ----------
        model : str
            The name of the ESM2 model to load. Must be one of the keys in 
            self.model2checkpoint.
        
        Returns
        -------
        model : transformers.AutoModel
            The loaded ESM2 model.
        tokenizer : transformers.AutoTokenizer
            The loaded ESM2 tokenizer.
        '''
        # Validate model
        if model not in self.model2checkpoint:
            raise ValueError(f'Invalid model: {model}')

        # Load the model and tokenizer
        checkpoint = self.model2checkpoint[model]
        model = transformers.AutoModel.from_pretrained(checkpoint)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint,
            clean_up_tokenization_spaces=False # avoids warning
        )

        # Disables dropout for deterministic results
        model.eval()

        # GPU/CPU
        model = model.cuda()

        return model, tokenizer

    def run(self, seq: str) -> None:
        '''
        Runs the ESM2 model on a single protein sequence.

        Parameters
        ----------
        seq : str
            The protein sequence to run the model on.
        '''
        # Validation
        if isinstance(seq, list):
            raise ValueError("Single sequence!")
        if len(seq) > 1400:
            raise ValueError("Too large of a protein!")
        tokens = re.findall(r'<[^>]+>|.', seq)
        if not all(token in self.tokenizer.get_vocab() for token in tokens):
            raise ValueError("Invalid amino acid(s) in sequence!")

        # Tokenization
        inputs = self.tokenizer(
            seq, 
            return_tensors='pt', 
            padding=True,
            truncation=False,
        )
        
        # GPU/CPU
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        # Run the model
        with torch.no_grad():
            self.output = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
    
    def representation(
        self, 
        layer: int = -1,
        special_tokens: bool = False,
        per: Literal['residue', 'sequence'] = 'residue'
    ) -> torch.Tensor:        
        '''
        Extracts the representation from a specific layer of the ESM2 model.

        Parameters
        ----------
        layer : int, optional
            The layer from which to extract the representation. 
            Default is -1 (last layer).
        special_tokens : bool, optional
            Whether to include special tokens (i.e., cls and eos) in the 
            representation. Default is False.
        per : {'residue', 'sequence'}, optional
            Whether to return the representation per residue or per sequence. 
            Default is 'residue'.

        Returns
        -------
        torch.Tensor
            The extracted representation.
        '''
        # Hidden state
        hidden_state = self.output.hidden_states[layer] # (batch, seq_len, hidden_size)

        # Remove special tokens if requested
        if not special_tokens:
            hidden_state = hidden_state[0][1:-1]
        else:
            hidden_state = hidden_state[0]

        # Per-residue or per-sequence
        if per == 'residue':
            return hidden_state.cpu().numpy()
        elif per == 'sequence':
            return hidden_state.mean(dim=0).cpu().numpy()
        else:
            raise ValueError("Invalid value for 'per'. Must be 'residue' or 'sequence'.")

if __name__ == "__main__":
    esm2 = ESM2('8M')
    esm2.run("KTAYIAKQRQISFVKSHFSRQDILDLWYHTQGYFPDWQNYTPGPGIRYPLKF")
    embedding = esm2.representation(layer=-1, per='residue')
