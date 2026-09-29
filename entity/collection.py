"""
===============================================================================
Title:      Collection
Outline:    Collection class that represents a collection of entities such as
            Proteins, PPIs, etc. It hosts functionalities that concern 
            collections as a whole and not individual entities.
            + ProteinCollection:
              - Load and save Protein objects from/to JSON files.
              - Iterate over Protein objects in the collection.
            + PPICollection:
              - Load and save PPI objects from/to JSON files.
              - Iterate over PPI objects in the collection.
Author:     Alejandro Sánchez Cano
Date:       29/09/2026
===============================================================================
"""

# Built-in modules
import json
from pathlib import Path
from dataclasses import asdict
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, Iterable

# Third-party modules
import numpy as np
from tqdm import tqdm

# Custom modules
from misc import paths
from .protein import Protein
from .pair import ProteinPair

class Collection(ABC):

###############################################################################
#########                     ABSTRACT METHODS                        #########
###############################################################################

    @abstractmethod
    def __init__(
        self, 
        file_path: str | Path | None = None, 
        items: list | None = None
        ):
        pass

    @abstractmethod
    def __iter__(self):
        pass 
    
    def __contains__(self, item) -> bool:
        return any(x == item for x in self)

###############################################################################
#########                     Protein Collection                      #########
###############################################################################

class ProteinCollection(Collection):

    def __init__(
        self,
        file_path: str | Path,
        proteins: Iterable[Protein] | None = None,
    ):
        self.file_path = Path(file_path)
        self.proteins = self._load() if proteins is None else proteins

    def _load(self) -> None:
        """Load Protein objects from the JSON file."""
        with self.file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [Protein(**protein) for protein in data]

    def __len__(self) -> int:
        """Number of Protein objects in the collection."""
        return len(self.proteins)

    def __iter__(self) -> Iterator[Protein]:
        """Iterate over Protein objects in the collection."""
        return iter(self.proteins)

    def save(self) -> None:
        """Save Protein objects to a JSON file."""
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [asdict(protein) for protein in self.proteins],
                handle,
                indent=2,
            )

###############################################################################
#########                    ProteinPair Collection                     #########
###############################################################################

class ProteinPairCollection(Collection):

    def __init__(
        self, 
        file_path: str | Path,
        pairs: Iterable[ProteinPair] | None = None,
        proteins: Iterable[Protein] | Path | str | None = None,
    ):
        self.file_path = Path(file_path)
        self.proteins = self._load_proteins(proteins)
        self.pairs = self._load() if pairs is None else pairs

    def _load_proteins(
        self, 
        proteins: Iterable[Protein] | Path | str | None
    ) -> list[Protein]:
        if proteins is None:
            return []
        if isinstance(proteins, (str, Path)):
            return ProteinCollection(file_path=proteins).proteins
        return list(proteins)

    def _load(self) -> list[ProteinPair]:
        """Load ProteinPair objects from the JSON file."""
        # Map UniProt accessions to Protein objects for quick lookup
        accession2protein = {
            protein.uniprot: protein 
            for protein in self.proteins
        }
        # Load ProteinPair objects from the JSON file
        with self.file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        return [
            ProteinPair(
                p1=accession2protein[pair["p1"]],
                p2=accession2protein[pair["p2"]],
                **{k: v for k, v in pair.items() if k not in ["p1", "p2"]}
            ) 
        for pair in data
        ]

    def __len__(self) -> int:
        """Number of ProteinPair objects in the collection."""
        return len(self.pairs)

    def __iter__(self) -> Iterator[ProteinPair]:
        """Iterate over ProteinPair objects in the collection."""
        return iter(self.pairs)
    
    def save(self) -> None:
        """Save ProteinPair objects to a JSON file."""
        # Substitute Protein objects with their UniProt accession
        for pair in self.pairs:
            pair.p1 = pair.p1.uniprot
            pair.p2 = pair.p2.uniprot
        
        # Save to JSON file
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [asdict(pair) for pair in self.pairs],
                handle,
                indent=2,
            )

if __name__ == "__main__":

    # Test the ProteinCollection class
    p1 = Protein(seq="MKRRL", taxon=9606, uniprot="P12345")
    p2 = Protein(seq="ACDEH", taxon=9606, uniprot="Q67890")
    p3 = Protein(seq="FGHIK", taxon=9606, uniprot="P54321")
    p4 = Protein(seq="LMNPY", taxon=9606, uniprot="Q98765")
    p5 = Protein(seq="QRSTV", taxon=9606, uniprot="P67890")
    p6 = Protein(seq="WGYZA", taxon=9606, uniprot="Q12345")
    proteins = [p1, p2, p3, p4, p5, p6]
    protein_path = paths.COLLECTIONS / 'mock.prot'
    collection = ProteinCollection(
        file_path=protein_path,
        proteins=proteins
    )
    for protein in collection:
        print(protein)
    collection.save()
    collection = ProteinCollection(file_path=protein_path)
    print(f"Loaded {len(collection)} proteins from {protein_path}")

    # Test the PPICollection class
    pair1 = ProteinPair(p1=p1, p2=p2, bind=1, mi_score=0.8)
    pair2 = ProteinPair(p1=p3, p2=p4, bind=0, mi_score=0.5)
    pair3 = ProteinPair(p1=p5, p2=p6, bind=1, mi_score=0.9)
    pair4 = ProteinPair(p1=p2, p2=p3, bind=0, mi_score=0.4)
    pair5 = ProteinPair(p1=p4, p2=p5, bind=1, mi_score=0.7)
    pair6 = ProteinPair(p1=p6, p2=p1, bind=0, mi_score=0.6)
    pair7 = ProteinPair(p1=p1, p2=p3, bind=1, mi_score=0.85)
    pair8 = ProteinPair(p1=p2, p2=p4, bind=0, mi_score=0.3)
    pairs = [pair1, pair2, pair3, pair4, pair5, pair6, pair7, pair8]
    pair_path = paths.COLLECTIONS / 'mock.pair'
    pair_collection = ProteinPairCollection(
        file_path=pair_path,
        pairs=pairs
    )
    for pair in pair_collection:
        print(pair)
    pair_collection.save()

    pair_collection = ProteinPairCollection(
        file_path=pair_path,
        proteins=protein_path
    )
    print(f"Loaded {len(pair_collection)} protein pairs from {pair_path}")
    print(pair_collection.pairs)
