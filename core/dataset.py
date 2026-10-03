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
from misc import config
from misc.logger import logger

class ProteinPairDataset(Dataset):

    def __init__(
        self, 
        pairs: list['ProteinPair'],
        tokenized: dict[str, tuple[torch.Tensor, torch.Tensor]] = None
    ):
        self.pairs = pairs
        self.tokenized = tokenized

    # FIXME: samples between two proteins of the the same of diferent main families?
    def log(self) -> None:
        # General sizes
        logger.info(f"Dataset size: {len(self.pairs)} pairs")
        logger.info(f"Positive pairs: {int(sum(pair.bind for pair in self.pairs))}")
        logger.info(f"Negative pairs: {int(len(self.pairs) - sum(pair.bind for pair in self.pairs))}")

        # Family sizes
        counts = {family: 0 for family in config.FAMILIES}
        for pair in self.pairs:
            if pair.p1.family in counts:
                counts[pair.p1.family] += 1
            if pair.p2.family in counts:
                counts[pair.p2.family] += 1
        logger.info("Family counts:")
        for family, count in counts.items():
            logger.info(f"  {family}: {count} pairs")
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        # Get pair
        pair = self.pairs[idx]

        # Return elements
        return {
            'input': (
                self.tokenized[pair.p1.uniprot],
                self.tokenized[pair.p2.uniprot]
            ),
            'label': torch.tensor(pair.bind, dtype=torch.float32),
            'identifier': (
                pair.p1.uniprot, 
                pair.p2.uniprot
            )
        }

if __name__ == "__main__":
    from entity.protein import Protein
    from entity.pair import ProteinPair
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345", family='PK_Tyr_Ser-Thr, Pkinase', architecture='PK_Tyr_Ser-Thr, Pkinase')
    p2 = Protein(seq="ACDE", taxon=9606, uniprot="Q67890", family='HSP70, MreB_Mbl', architecture='HSP70, MreB_Mbl')
    p3 = Protein(seq="FGHI", taxon=9606, uniprot="P54321", family='PK_Tyr_Ser-Thr, Pkinase', architecture='PK_Tyr_Ser-Thr, Pkinase')
    pair1 = ProteinPair(p1=p1, p2=p2, bind=1, mi_score=0.8)
    pair2 = ProteinPair(p1=p2, p2=p1, bind=1, mi_score=0.7)
    pair3 = ProteinPair(p1=p1, p2=p3, bind=0, mi_score=0.5)
    tokenized = {
        p1.uniprot: (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]])),
        p2.uniprot: (torch.tensor([[4, 5, 6]]), torch.tensor([[1, 1, 1]])),
        p3.uniprot: (torch.tensor([[7, 8, 9]]), torch.tensor([[1, 1, 1]])),
    }
    pairs = [pair1, pair2, pair3]
    dataset = ProteinPairDataset(pairs, tokenized=tokenized)
    for idx in range(len(dataset)):
        print(dataset[idx])
    #dataset.log()