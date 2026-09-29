"""
===============================================================================
Title:      Protein
Outline:    Protein dataclass that represents a protein entity.
            Attributes
            ----------
            - seq: Protein sequence (or file path to pickled Protein object).
            - uniprot: UniProt ID.
            - taxon: Taxon ID (e.g. 9606).
Author:     Alejandro Sánchez Cano
Date:       17/09/2026
===============================================================================
"""

# Built-in modules
import pprint
import hashlib
from dataclasses import dataclass, field

@dataclass(slots=True)
class Protein:
    seq: str = None
    uniprot: str = None
    taxon: int = None
    family: str = None
    architecture: str = None

    def __eq__(self, other: 'Protein') -> bool:
        return self.seq == other.seq and self.taxon == other.taxon

    def __hash__(self) -> int:
        hash_input = f"{self.seq}_{self.taxon}".encode('utf-8')
        return int(hashlib.md5(hash_input).hexdigest(), 16)

if __name__ == "__main__":
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345")
    p2 = Protein(seq="MK", taxon=9606, uniprot="Q67890")
    p3 = Protein(seq="ACDE", taxon=9606, uniprot="P54321", family=[["PF00001", "PF00002"]])
    prots = {p1, p2, p3}
    pprint.pprint(prots)