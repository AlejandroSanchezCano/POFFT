"""
===============================================================================
Title:      NegativeSampler
Outline:    Classes to sample negative protein pairs from a set of positive
            protein pairs.
            https://github.com/elenaliar/msc_thesis/blob/main/utils/negative_sampling.py#L83-L153
Docs:       https://github.com/daisybio/data-leakage-ppi-prediction/blob/main/create_gold_standard.py#L107
Author:     Alejandro Sánchez Cano
Date:       21/09/2026
===============================================================================
"""

# Built-in modules
import random
from collections import defaultdict

# Custom modules
from misc.logger import logger
from entity.protein import Protein
from entity.pair import ProteinPair

class NegativeSampler:

    def __init__(
        self,
        positive_pairs: list[ProteinPair],
        seed: int | None = 42,
    ):
        self.positive_pairs = set(positive_pairs)
        if len(self.positive_pairs) != len(positive_pairs):
            raise ValueError("Duplicate pairs found in positive_pairs.")
        self.negative_pairs = set()
        self.rng = random.Random(seed)

    def _degree(self) -> dict[Protein, int]:
        """
        Calculate the degree of each protein in the positive pairs.
        As protein interaction networks (PINs) are undirected graphs, a self
        loop (a protein interacting with itself) contributes 2 to the degree of
        that protein.

        Returns
        -------
        dict[Protein, int]
            A dictionary mapping each Protein object to its degree (number of
            interactions) in the positive pairs.
        """
        degree = defaultdict(int)
        for pair in self.positive_pairs:
            degree[pair.p1] += 1
            degree[pair.p2] += 1
        return degree

    def _sample(self) -> ProteinPair:
        """
        Use random.choices to sample two proteins (with replacement) based on
        their degree in the positive pairs. This ensures that proteins with 
        higher degrees are more likely to be sampled.

        Returns
        -------
        ProteinPair
            A randomly sampled negative ProteinPair object.
        """
        degree_dict = self._degree()
        pair = self.rng.choices(
            population=list(degree_dict.keys()),
            weights=list(degree_dict.values()),
            k=2
        )
        return ProteinPair(
            p1=pair[0], 
            p2=pair[1], 
            bind=0, 
            mi_score=None
        )

    def _reject(self, pair: ProteinPair) -> bool:
        """
        Check if the sampled pair is a positive pair, a seen negative pair, or 
        a homodimer.

        Returns
        -------
        bool
            Whether to reject or preserve the sampled pair.
        """
        return (
            pair in self.positive_pairs or 
            pair in self.negative_pairs or
            pair.p1 == pair.p2
        )

    def _validate_ratio(self):
        pass

    def sample(self, ratio: int = 10) -> list[ProteinPair]:
        """
        Sample negative pairs until the desired ratio of negative/positive
        pairs is reached.

        Parameters
        ----------
        ratio : int, optional
            The desired ratio of negative to positive pairs, by default 10.

        Returns
        -------
        list[ProteinPair]
            A list of sampled negative ProteinPair objects.
        """
        # Validate ratio
        total_proteins = len(self._degree())
        total_pairs = total_proteins * (total_proteins - 1) // 2
        homodimers = sum(1 for pair in self.positive_pairs if pair.p1 == pair.p2)
        max_negatives = total_pairs - len(self.positive_pairs) + homodimers
        requested_negatives = len(self.positive_pairs) * ratio
        if requested_negatives > max_negatives:
            raise ValueError(
                f"Requested ratio of {ratio} corresponds to "
                f"a requested number of {requested_negatives} pairs, but the "
                f"maximum possible number of negative pairs is {max_negatives}. "
                f"Please choose a lower ratio."
            )

        # Sample negative pairs
        while len(self.negative_pairs) / len(self.positive_pairs) < ratio:
            pair = self._sample()
            if not self._reject(pair):
                logger.debug(f"Accepted pair: {pair.p1.uniprot}-{pair.p2.uniprot}")
                self.negative_pairs.add(pair)
            else:
                logger.debug(f"Rejected pair: {pair.p1.uniprot}-{pair.p2.uniprot}")
        return list(self.negative_pairs)

if __name__ == "__main__":
    # Example usage
    p1 = Protein(seq="MK", taxon=9606, uniprot="P12345")
    p2 = Protein(seq="ACDE", taxon=9606, uniprot="Q67890")
    p3 = Protein(seq="FGHI", taxon=9606, uniprot="P54321")
    p4 = Protein(seq="LMNOP", taxon=9606, uniprot="R98765")

    pair1 = ProteinPair(p1=p1, p2=p2, bind=1, mi_score=0.8)
    pair2 = ProteinPair(p1=p2, p2=p3, bind=1, mi_score=0.7)
    pair3 = ProteinPair(p1=p1, p2=p4, bind=1, mi_score=0.5)
    pair4 = ProteinPair(p1=p4, p2=p4, bind=1, mi_score=0.9)

    positive_pairs = [pair1, pair2, pair3, pair4]
    sampler = NegativeSampler(positive_pairs=positive_pairs)
    degree_dict = sampler._degree()
    print(degree_dict)
    print('\n')
    print(sampler._sample())
    print('\n')
    print(sampler.sample())