"""
===============================================================================
Title:      ProteinPair
Outline:    ProteinPair dataclass that represents a protein-protein interaction.
            Attributes
            ----------
            - p1: Protein object for the first protein.
            - p2: Protein object for the second protein.
            - bind: Binary indicator for binding (1 if binding, 0 if not).
            - mi_score: MI-score for the interaction from IntAct.
Author:     Alejandro Sánchez Cano
Date:       18/09/2026
===============================================================================
"""

# Built-in modules
import pprint
from typing import Any
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

# Custom modules
from .protein import Protein

@dataclass(slots=True)
class ProteinPair:
    p1: Protein = None
    p2: Protein = None
    bind: int = None
    mi_score: float = None

    def __eq__(self, other: 'ProteinPair') -> bool:
        AABB = (self.p1 == other.p1 and self.p2 == other.p2)
        ABAB = (self.p1 == other.p2 and self.p2 == other.p1)
        return AABB or ABAB

    def __hash__(self) -> int:
        protein_set = frozenset([self.p1, self.p2])
        return hash(protein_set)

if __name__ == "__main__":
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345")
    p2 = Protein(seq="ACDE", taxon=9606, uniprot="Q67890")
    p3 = Protein(seq="FGHI", taxon=9606, uniprot="P54321")
    pair1 = ProteinPair(p1=p1, p2=p2, bind=1, mi_score=0.8)
    pair2 = ProteinPair(p1=p2, p2=p1, bind=1, mi_score=0.7)
    pair3 = ProteinPair(p1=p1, p2=p3, bind=0, mi_score=0.5)
    pairs = {pair1, pair2, pair3}
    pprint.pprint(pairs)