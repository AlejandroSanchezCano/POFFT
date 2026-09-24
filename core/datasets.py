"""
===============================================================================
Title:      Datasets
Outline:    Functions to create and manage datasets for deep learning models:
            - ProteinDataset: Dataset for individual proteins.
            - ProteinPairDataset: Dataset for protein pairs.
Author:     Alejandro Sánchez Cano
Date:       23/09/2026
===============================================================================
"""

# Third-party modules
import torch
from torch.utils.data import Dataset

# Custom modules
from misc.logger import logger

class ProteinDataset(Dataset):

    def __init__(self, proteins: list['Protein']):
        self.proteins = proteins

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.proteins[idx].seq,
            None,  # Placeholder for label
            self.proteins[idx].uniprot
        )

class ProteinPairDataset(Dataset):

    def __init__(self, pairs: list['ProteinPair']):
        self.pairs = pairs

    def log(self) -> None:
        logger.info(f"Dataset size: {len(self.pairs)} pairs")
        logger.info(f"Positive pairs: {int(sum(pair.bind for pair in self.pairs))}")
        logger.info(f"Negative pairs: {int(len(self.pairs) - sum(pair.bind for pair in self.pairs))}")

    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> tuple:
        return (
            self.pairs[idx].p1.seq,
            self.pairs[idx].p2.seq,
            torch.tensor(self.pairs[idx].bind, dtype=torch.float32),
            (
                self.pairs[idx].p1.uniprot, 
                self.pairs[idx].p2.uniprot
            )
        )

if __name__ == "__main__":
    from entity.protein import Protein
    from entity.pair import ProteinPair
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345")
    p2 = Protein(seq="ACDE", taxon=9606, uniprot="Q67890")
    p3 = Protein(seq="FGHI", taxon=9606, uniprot="P54321")
    pair1 = ProteinPair(p1=p1, p2=p2, bind=1, mi_score=0.8)
    pair2 = ProteinPair(p1=p2, p2=p1, bind=1, mi_score=0.7)
    pair3 = ProteinPair(p1=p1, p2=p3, bind=0, mi_score=0.5)
    pairs = [pair1, pair2, pair3]
    dataset = ProteinPairDataset(pairs)
    for idx in range(len(dataset)):
        print(dataset[idx])