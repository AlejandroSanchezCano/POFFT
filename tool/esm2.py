"""
===============================================================================
Title:      ESM2
Outline:    ESM2 class to interact with META's ESM2 model. It supports:
            - Loading different ESM2 models
            - Preparing data: validation and batch conversion
            - Running the model for specific layers
            - Estimating perplexity (fast and pseudo)
            - Extracting representations (per-residue and per-sequence)
            - Plotting attention maps
            - Getting contact maps
Docs:       https://github.com/facebookresearch/esm
Author:     Alejandro Sánchez Cano
Date:       22/09/2026
===============================================================================
"""

# Built-in libraries
from typing import Literal

# Third-party libraries
import esm
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ESM2:

    model2name = {
        '8M' : 'esm2_t6_8M_UR50D',
        '35M' : 'esm2_t12_35M_UR50D',
        '150M' : 'esm2_t30_150M_UR50D',
        '650M' : 'esm2_t33_650M_UR50D',
        '3B' : 'esm2_t36_3B_UR50D',
        '15B' : 'esm2_t48_15B_UR50D'
    }

    def __init__(self, model: str):
        self.model_name = self.model2name[model] # esm2_t33_650M_UR50D
        self.num_parameters = int(model[:-1]) * 10**6 if model[-1] == 'M' else 10**9 # 650M
        self.num_layers = int(self.model_name.split('_')[1][1:]) # 33
        self.model, self.alphabet = self._load()
        self.batch = None
        self.output = None

    def _load(self) -> tuple[esm.model.esm2.ESM2, esm.data.Alphabet]:
            '''
            Load ESM2 model and alphabet.

            Returns
            -------
            tuple[esm.model.esm2.ESM2, esm.data.Alphabet]
                ESM2 model and alphabet.
            '''

            # Load model and alphabet
            model, alphabet = eval(f'esm.pretrained.{self.model_name}()')

            # Disables dropout for deterministic results
            model.eval()

            # GPU/CPU
            model = model.cuda()

            # Logger
            print('Model is loaded')

            return model, alphabet

    def prepare_data(self, data: list[tuple[str, str]]) -> None:
        '''
        Prepare data for embedding using alphabet.get_batch_converter() method.

        Parameters
        ----------
        data : list[tuple[str, str]]
            List of tuples containing the protein label (id) and the protein 
            sequence. It is recommended to process one protein at a time to 
            avoid memory issues.
        '''
        # Validate input
        if len(data) != 1: raise ValueError('Only one protein should be processed at a time')
        if len(data[0][1]) > 1400: raise ValueError('Too big of a protein, this would cause a torch.cuda.OutOfMemoryError')
        possible_residues = set(self.alphabet.standard_toks)
        protein_residues = set(''.join([seq for _, seq in data]))
        if not protein_residues.issubset(possible_residues):
            invalid_residues = protein_residues - possible_residues
            raise ValueError(f'Sequence contains non-standard amino acid tokens: {invalid_residues}\n Allowed residues: {possible_residues}')
        
        # Batch conversion
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        # Custom IDs, sequences, tokens, and lengths
        self.batch = {
            'ids' : batch_labels,
            'seqs' : batch_strs,
            'tokens' : batch_tokens,
            'lens' : batch_lens
        }

    def run_model(self, layers: list = None) -> dict[Literal['logits', 'representations', 'attentions', 'contacts'], torch.Tensor]:
        '''
        Run ESM2 for specific layers.

        Parameters
        ----------
        layers : list
            List of layer indices to extract from. Default is the last layer.

        Returns
        -------
        dict[Literal['logits', 'representations', 'attentions', 'contacts'], torch.Tensor]
            Dictionary containing the logits, representations, attentions, and
            contacts results of the model.
        '''
        
        # Run model
        layers = layers if layers is not None else [self.num_layers]
        with torch.no_grad():
            output = self.model(self.batch['tokens'].cuda(), repr_layers = layers, return_contacts = True)
        
        # Shape sanity checks
        tokens_length = len(self.batch['tokens'][0])
        sequences_length = len(self.batch['seqs'][0])
        assert tokens_length == sequences_length + 2 # because <cls> and <eos> tokens
        #assert list(output['logits'].shape) == [1, tokens_length, len(self.alphabet)]
        #assert list(output['representations'][self.num_layers].shape) == [1, tokens_length, 2560]
        #assert list(output['attentions'].shape) == [1, self.num_layers, 20, tokens_length, tokens_length]
        #assert list(output['contacts'].shape) == [1, sequences_length, sequences_length]
        # 1280 is embedding dimension
        # 20 is number of attention heads (?)

        # Remove first dimension because batch size will always be 1
        output['logits'] = output['logits'][0]
        output['attentions'] = output['attentions'][0]
        output['contacts'] = output['contacts'][0]

        # Store logits, representations, attentions, and contacts
        self.output = output

        return output

    def fast_perplexity(self) -> float:
        '''
        Estimate perplexity of the model on a given protein sequence by 
        calculating the exponential of the negative log-likelihood of the 
        tokens in the mask set M. Each token has a 15% probability of inclusion 
        in the mask set M. If included the tokens have an 80% probability of 
        being replaced with a mask token, a 10% probability of being replaced 
        with a random token, and a 10% probability of being replaced with an 
        unmasked token. The perplexity is calculated using the logits (outputted
        by the last layer) of the tokens in the mask set M conditional on the 
        unmodified tokens.
        It is a stochastic approach, but requires only one forward pass.

        Returns
        -------
        float
            Perplexity of the model on the protein sequence.
        '''
        
        # Instantiate variables
        perplexity = 0
        idx_in_masked_set = []

        # Clone token tensor
        tokens = self.batch['tokens'].clone()

        # Iterate over sequence indices
        for seq_idx in range(1, self.batch['lens'][0] - 1):
            
            # 15% chance of being in the "masked" set 
            if torch.rand(1).item() < 0.15:
                idx_in_masked_set.append(seq_idx)
                
                # Apply the masking strategy (80% mask token, 10% random token, 10% unchanged)
                random_value = torch.rand(1).item()
                if random_value < 0.8:
                    tokens[0, seq_idx] = self.alphabet.mask_idx
                elif random_value < 0.9:
                    tokens[0, seq_idx] = torch.randint(4, 24, (1,)).item()
                else:
                    pass
        
        # Run model
        with torch.no_grad():
                output = self.model(tokens.cuda(), repr_layers = [self.num_layers], return_contacts = False)
        
        # Calculate log-probabilities for the tokens in "masked" set
        logits = output['logits'][0]
        for masked_idx in idx_in_masked_set:
            log_probs = torch.nn.functional.log_softmax(logits[masked_idx, 4:24], dim = -1)
            true_token_idx = self.batch['tokens'][0, masked_idx] - 4
            log_prob = log_probs[true_token_idx].item()
            perplexity += log_prob
        
        # Indifinte perplexity if no tokens in "masked" set
        if len(idx_in_masked_set) == 0: return float('inf') 

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(-perplexity/len(idx_in_masked_set)))

        return perplexity.item()

    def pseudo_perplexity(self) -> float:
        '''
        Estimate perplexity as the pseudo-perplexity of the model on a given 
        sequence, which is calculated as the exponential of the negative 
        log-likelihood of a sequence divided by its length using the logits 
        outputted by the last layer. It's a deterministic approach, but 
        requires multiple forward passes to mask each token in the sequence.
        Special tokens like <cls>, <eos>, <pad>, <mask>, <unk>, as well as 
        non-standard aminoacid tokens like F, L, J, O, U, X, Z are excluded
        from the calculation.

        TODO: The results don't match the paper for ESM2 >= 650M, so it needs to be fixed.

        Returns
        -------
        float
            Pseudo perplexity of the protein sequence.
        '''
        # Instantiate pseudo perplexity
        pseudo_perplexity = 0

        # Iterate over sequence indices
        for seq_idx in tqdm(range(1, self.batch['lens'][0] - 1)):
            # Mask ith token
            tokens = self.batch['tokens'].clone()
            tokens[0, seq_idx] = self.alphabet.mask_idx

            # Run model
            with torch.no_grad():
                output = self.model(tokens.cuda(), repr_layers = [self.num_layers], return_contacts = False)
            
            # Calculate pseudo perplexity for ith token from logits of standard amino acids (token indeces 4 to 23)
            logits = output['logits'][0]
            log_likelihood = torch.nn.functional.log_softmax(logits[seq_idx, 4:24], dim = -1)
            true_token_idx = self.batch['tokens'][0, seq_idx] - 4
            pseudo_log_likelihood = log_likelihood[true_token_idx].item()
            pseudo_perplexity += pseudo_log_likelihood
        
        seq_len = self.batch['lens'][0] - 2
        pseudo_perplexity = torch.exp(-pseudo_perplexity/seq_len)
        return pseudo_perplexity.item()

    def extract_representations(self, layer: int = None) -> tuple[np.ndarray, np.ndarray]:
        '''
        Extract representations/embeddings from a model's layer.
        It excludes the <cls> and <eos> tokens.

        Parameters
        ----------
        layer : int
            Layer index to extract representations from. Default is the last 
            layer.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple containing the per-residue and per-sequence representations.
        '''
        layer = layer if layer is not None else self.num_layers
        representations = self.output['representations'][layer][0][1:-1]
        per_residue = representations.cpu().numpy()
        per_sequence = representations.mean(0).cpu().numpy()

        return per_residue, per_sequence

    def plot_attention_map(self, layer: int) -> None:
        '''
        Plot the attention map for a given layer by averaging the 
        attention maps over all heads. 
        It excludes the <cls> and <eos> tokens.

        Parameters
        ----------
        layer : int
            Layer index to plot the attention map for.
        '''
        # Extract attention map
        attentions = self.output['attentions'][layer - 1].mean(0)[1:-1, 1:-1]
        return attentions

        # Plot
        plt.figure(figsize = (5, 5))
        plt.imshow(attentions.cpu().numpy(), cmap = 'magma')
        plt.colorbar()
        plt.title(f'Layer {layer} Attention Map')
        #plt.show()
        plt.savefig(f'{layer}_attention_map.png')
        plt.close()

    def contact_map(self) -> np.ndarray:
        '''
        Get contact map.

        Returns
        -------
        np.ndarray
            Contact map of the protein sequence.
        '''
        contacts = self.output['contacts']
        return contacts.cpu().numpy()

if __name__ == '__main__':
    
    data = [('id1', 'TRKVAIYGKGGIGKSTTTQNTAAALAYFHDKKVFIHGCDPKADSTRLILGGKPQETLMDMLRDKGAEKITNDDVIKKGFLDIQCVESGGPEPGVGCAGRGVITAIDLMEENGAYTDDLDFVFFDVLGDVVCGGFAMPIRDGKAQEVYIVASGEMMAIYAANNICKGLVKYAKQSGVRLGGIICNSRKVDGEREFLEEFTAAIGTKMIHFVPRDNIVQKAEFNKKTVTEFAPEENQAKEYGELARKIIENDEFVIPKPLTMDQLEDMVVKYGIAD')]

    esm2 = ESM2('650M')
    esm2.prepare_data(data)
    esm2.run_model(layers = [27, 28, 29, 30, 31, 32, 33])
    r, s = esm2.extract_representations(layer = 33)
    esm2.plot_attention_map(layer = 33)
    print(len(esm2.contact_map())==len(data[0][1]))
    print(len(esm2.plot_attention_map(layer = 33)))
    print(len(data[0][1]), r.shape, s.shape)
