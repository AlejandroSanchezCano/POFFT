"""
===============================================================================
Title:      Build collections
Outline:    Load the necessary data and construct a list of all Protein objects
            in the fasta files. Use it to go over the interaction dataset and
            create a list of Protein and ProteinPair objects. Sample negative
            pairs, create collection objects and save them.
Author:     Alejandro Sánchez Cano
Date:       28/09/2026
Time:       10 min
===============================================================================
"""

# Built-in modules
import re
import json
import random

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from misc import paths
from misc import config
from tool.fasta import Fasta
from misc.logger import logger
from entity.protein import Protein
from entity.pair import ProteinPair
from negative_sampler import NegativeSampler
from cluster.domain_architecture import DomainArchitecture
from entity.collection import ProteinCollection, ProteinPairCollection
logger.info('Importing modules completed')
logger.setLevel('INFO')

###############################################################################
#######                           LOAD DATA                             #######
###############################################################################

# Load fasta
fasta_file = paths.CLUSTER / 'sequences.fasta'
fasta = Fasta.from_file(fasta_file)

# Load clustering
json_file = paths.CLUSTER / 'accession2representative.json'
with open(json_file, 'r') as handle:
    accession2representative = json.load(handle)

# Load architectures
json_file = paths.CLUSTER / 'accession2architecture.json'
with open(json_file, 'r') as handle:
    accession2architecture = json.load(handle)

# Load interactions
logger.info('Loading interaction data...')
interactions_file = paths.CLUSTER / 'interactions.txt'
df = pd.read_csv(interactions_file, sep='\t')

###############################################################################
#######                     ALL PROTEINS COLLECTION                     #######
###############################################################################

# Create Protein objects
accession2protein = {}
for header, seq in tqdm(fasta.records, desc='Creating Protein objects'):
    # Obtain protein information
    uniprot = header.split('|')[1]
    taxon = int(re.search(r'OX=([0-9]+)', header).group(1))
    family = accession2representative[uniprot]
    family = DomainArchitecture(family).prettify()
    architecture = accession2architecture[uniprot]
    architecture = DomainArchitecture(architecture).prettify()
    # Separate 'Actin' family on the fly
    if family == 'Actin' and architecture != 'Actin':
        family = 'HSP70, MreB_Mbl'
    # Create Protein object
    protein = Protein(
        uniprot=uniprot, 
        seq=seq, 
        taxon=taxon, 
        family=family,
        architecture=architecture
    )
    accession2protein[uniprot] = protein
logger.info(f"Number of total proteins: {len(accession2protein)}")

###############################################################################
#######                      FAMILIES COLLECTIONS                       #######
###############################################################################

# Previously we had used a 'proteins' set to store the Protein objects found
# in the interaction dataset. However, this had the drawback that some two
# UniProt accessions could point to the same Protein object (e.g. in the case 
# of P01116-1 and P01116). Instead of putting them together, which I found a 
# hassle, we keep an 'accessions' set and later recover the Protein objects 
# from it using the 'accession2protein' dictionary. Thisq way, we manage to
# avoid duplicates while keeping the code clean. However, this means that we 
# have itnroduce Protein objects that are technically the same (same sequence
# and same taxon) but have different UniProt accessions!!!!!!!!!!!!!!!!!!!!!!!!

# Create Protein objects
pairs = set()
accessions = set()
for row in tqdm(df.itertuples(index=False, name=None), total=len(df)):
    # Obtain proteins
    protein_A = accession2protein[row[0]]
    protein_B = accession2protein[row[1]]
    # Add to sets if one protein belongs to the selected families
    A_in_families = protein_A.family in config.FAMILIES
    B_in_families = protein_B.family in config.FAMILIES
    if A_in_families or B_in_families:
        # Add accessions
        accessions.add(row[0])
        accessions.add(row[1])
        # Add pair
        pair = ProteinPair(
            p1=protein_A,
            p2=protein_B,
            bind=1
        )
        pairs.add(pair)

# Convert to lists
pairs = list(pairs)
proteins = [accession2protein[acc] for acc in accessions]

# Logging
logger.info(f"Number of proteins in selected families: {len(proteins)}")
logger.info(f"Number of pairs in selected families: {len(pairs)}")

# Create ProteinCollection
protein_path = paths.COLLECTIONS / 'families.prot'
collection = ProteinCollection(
    file_path=protein_path,
    proteins=proteins
)
collection.save()

# Sample negative pairs
sampler = NegativeSampler(positive_pairs=pairs)
negative_pairs = sampler.sample(ratio=config.NEGATIVE_TO_POSITIVE_RATIO)
logger.info(f"Number of sampled negative pairs: {len(negative_pairs)}")

# Shuffle pairs
random.shuffle(pairs)
random.shuffle(negative_pairs)

# Create ProteinPairCollection
pair_path = paths.COLLECTIONS / 'families.pair'
collection = ProteinPairCollection(
    file_path=pair_path,
    proteins=proteins,
    pairs=pairs + negative_pairs
)
collection.save()