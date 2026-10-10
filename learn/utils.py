"""
===============================================================================
Title:      Utils
Outline:    Utility functions for the training scripts:
            - extract_pairs: gets protein pairs from a subset of a dataset.
            - make_collate_fn: creates a Dataloader collate function
Author:     Alejandro Sánchez Cano
Date:       08/10/2026
===============================================================================
"""

# Third-party modules
import torch
from transformers import DataCollatorWithPadding

def extract_pairs(
    subset: torch.utils.data.dataset.Subset
) -> list['ProteinPair']:
    '''
    Extracts the protein pairs from a subset of a dataset.

    Parameters
    ----------
    subset : torch.utils.data.dataset.Subset
        A subset of a dataset containing protein pairs.

    Returns
    -------
    list['ProteinPair']
        A list of protein pairs extracted from the subset.
    '''
    return [subset.dataset.pairs[idx] for idx in subset.indices]

def make_collate_fn(tokenizer) -> callable:
    '''
    Creates a collate function for the DataLoader that uses the provided 
    tokenizer (e.g., ESM2's tokenizer), allowing for dynamic padding of 
    sequences in a batch.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for padding sequences.

    Returns
    -------
    callable
        A collate function that can be used with a DataLoader.
    '''
    # Dynamic padding
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )

    # Collate function
    def collate_fn(batch: list[dict]) -> dict:
        
        # Access elements
        tokenized = [item['input'] for item in batch]
        labels = [item['label'] for item in batch]
        identifiers = [item['identifier'] for item in batch]
        tokenized1, tokenized2 = zip(*tokenized)

        # Convert to specific format for collator: 
        # {
        #     'input_ids': (seq_len,),
        #     'attention_mask': (seq_len,)
        # }
        batch1 = [
            {
                'input_ids': input_ids.squeeze(0), 
                'attention_mask': attention_mask.squeeze(0)
            }
            for input_ids, attention_mask in tokenized1 # (1, seq_len)
        ]
        batch2 = [
            {
                'input_ids': input_ids.squeeze(0), 
                'attention_mask': attention_mask.squeeze(0)
            }
            for input_ids, attention_mask in tokenized2 # (1, seq_len)
        ]

        # Collate batches
        collated1 = collator(batch1)
        collated2 = collator(batch2)

        return {
            'inputs': (
                (collated1['input_ids'], collated1['attention_mask']),
                (collated2['input_ids'], collated2['attention_mask'])
            ),
            'labels': torch.stack(labels),
            'identifiers': identifiers
        }

    return collate_fn