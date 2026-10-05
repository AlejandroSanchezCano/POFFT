"""
===============================================================================
Title:      LengthBatchSampler
Outline:    Custom batch sampler for variable-length sequences that groups 
            sequences of similar lengths into buckets to minimize padding and 
            improve training efficiency.
Author:     Alejandro Sánchez Cano
Date:       05/10/2026
===============================================================================
"""

# Built-in modules
import random

# Third-party mofules
from torch.utils.data import Dataset, Sampler

class LengthBatchSampler(Sampler):

    def __init__(
        self, 
        data_source: Dataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.data_source = data_source
        self.sorted_indices = sorted(
            range(len(data_source)), 
            key=lambda idx: self._get_length(idx)
        )
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _get_length(self, idx: int) -> int:
        '''
        Get the length of the protein pair at the given index in the dataset,
        defined as the combined length of the two proteins in the pair. The 
        implementation of this method depends on the structure of the dataset
        and how the protein sequences are stored.

        Parameters
        ----------
        idx : int
            The index of the protein pair in the dataset.

        Returns
        -------
        int
            The combined length of the two proteins in the pair.
        '''
        tokenized1, tokenized2 = self.data_source[idx]['input']
        return tokenized1[0].numel() + tokenized2[0].numel()

    def __len__(self) -> int:
        '''
        How many batches will __iter__ yield?

        Returns
        -------
        int
            The number of batches in the dataset.
        '''
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        '''
        Iterate over the length-sorted indices of the dataset, yielding batches
        of indices that correspond to sequences of similar lengths. If 
        shuffling is enabled, the order of the batches and the order of the
        indices within each batch will be randomized.

        Yields
        ------
        list
            A list of indices corresponding to a batch of sequences of similar 
            lengths.
        '''
        # Batch the sorted indices
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]

        # Shuffle
        # TODO: control randomness with random.Random()
        if self.shuffle:
            random.shuffle(batches)
            for batch in batches:
                random.shuffle(batch)

        # Yield batches
        yield from batches

if __name__ == "__main__":
    # Example usage
    import torch
    from core.dataset import ProteinPairDataset
    from torch.utils.data import DataLoader
    from entity.protein import Protein
    from entity.pair import ProteinPair
    
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345", family='PK_Tyr_Ser-Thr, Pkinase', architecture='PK_Tyr_Ser-Thr, Pkinase')
    p2 = Protein(seq="ACDEFF", taxon=9606, uniprot="Q67890", family='HSP70, MreB_Mbl', architecture='HSP70, MreB_Mbl')
    p3 = Protein(seq="FG", taxon=9606, uniprot="P54321", family='PK_Tyr_Ser-Thr, Pkinase', architecture='PK_Tyr_Ser-Thr, Pkinase')
    pair1 = ProteinPair(p1=p1, p2=p2, bind=1, mi_score=0.8)
    pair2 = ProteinPair(p1=p2, p2=p1, bind=1, mi_score=0.7)
    pair3 = ProteinPair(p1=p1, p2=p3, bind=0, mi_score=0.5)
    pair4 = ProteinPair(p1=p3, p2=p2, bind=0, mi_score=0.4)
    pairs = [pair1, pair2, pair3, pair4]
    tokenized = {
        p1.uniprot: (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]])),
        p2.uniprot: (torch.tensor([[4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 1, 1, 1, 1, 1]])),
        p3.uniprot: (torch.tensor([[10, 11]]), torch.tensor([[1, 1]])),
    }
    dataset = ProteinPairDataset(pairs, tokenized=tokenized)
    sampler = LengthBatchSampler(dataset, batch_size=2, shuffle=True)
    print(f'Sorted indices: {sampler.sorted_indices}')
    for batch_indices in sampler:
        print(f"Batch indices: {batch_indices}")
        for idx in batch_indices:
            print(dataset[idx])