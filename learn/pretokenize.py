"""
===============================================================================
Title:      Pretokenize
Outline:    This script pre-tokenizes protein sequences using the ESM2 model
            and saves the tokenized sequences to a file. This is useful for
            speeding up training and inference by avoiding repeated 
            tokenization of the same sequences.
Author:     Alejandro Sánchez Cano
Date:       07/10/2026
Time:       1 min
===============================================================================
"""

# Third-party modules
import torch
from tqdm import tqdm

# Custom modules
from misc import paths
from misc import config
from tool.esm2 import ESM2
from misc.logger import logger
from entity.collection import ProteinPairCollection
logger.info('Importing modules completed')

# Load protein pairs
protein_path = paths.COLLECTIONS / 'families.prot'
file_path = paths.COLLECTIONS / 'families.pair'
collection = ProteinPairCollection(
    file_path=file_path,
    proteins=protein_path,
)

# Initialize ESM2
esm2 = ESM2(config.ESM2_MODEL)

# Tokenize sequences
tokenized = {
    protein.uniprot: esm2.tokenize(protein.seq)
    for protein in tqdm(collection.proteins, desc="Tokenizing proteins")
}

# Save tokenized sequences
tokenized_path = paths.COLLECTIONS / 'families.tokenized'
torch.save(tokenized, tokenized_path)