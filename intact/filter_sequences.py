"""
===============================================================================
Title:      Filter sequences by length
Outline:    Compares the UniProt accessions obtained from fetching their 
            sequences with the accessions present in the IntAct dataset
            and keeps the intersecting interactors and their corresponding
            interactions. Then, it filters the sequences by length < 800 amino
            acids.
Author:     Alejandro Sánchez Cano
Date:       29/08/2026
Time:       2 min
===============================================================================
"""

# Built-in modules
import os
import json

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from misc import paths
from misc import config
from fasta import Fasta
from misc.logger import logger
logger.info('Importing modules completed')

###############################################################################
#######                          MAP ACCESSIONS                         #######
###############################################################################

# Load dataframe
df = pd.read_csv(
    paths.INTACT / config.INTACT_VERSION / 'uniprot_nr.txt',
    sep='\t',
)

# Load accession mapping
mapper_path = paths.INTACT / config.INTACT_VERSION / 'accession_mapping.json'
with open(mapper_path, 'r') as handle:
    accession_mapping = json.load(handle)

# Map
logger.info(f'Mapping accessions...')
mapping = lambda x: accession_mapping.get(x, x)
df['#ID(s) interactor A'] = df['#ID(s) interactor A'].apply(mapping)
df['ID(s) interactor B'] = df['ID(s) interactor B'].apply(mapping)

###############################################################################
#######              DISREGARD ACCESSIONS WITHOUT SEQUENCE              #######
###############################################################################

# UniProt accessions from headers
fasta_path = paths.INTACT / config.INTACT_VERSION / 'sequences.fasta'
fasta = Fasta.from_file(fasta_path)
fasta_accessions = [header.split('|')[1] for header in fasta.headers]
logger.info(f'Total UniProt accessions in fasta file: {len(fasta_accessions)}')
fasta_accessions = set(fasta_accessions)
logger.info(f'Unique UniProt accessions in fasta file: {len(fasta_accessions)}')

# Remove rows without fetched sequences
has_fetched_sequence = lambda x: x in fasta_accessions
filtered_df = df[
        (df['#ID(s) interactor A'].apply(has_fetched_sequence)) &
        (df['ID(s) interactor B'].apply(has_fetched_sequence))
    ]
filtered_accessions = set(pd.concat([
    filtered_df['#ID(s) interactor A'],
    filtered_df['ID(s) interactor B']
]).unique())

# Filter fasta records to only include those matching in the filtered DataFrame
matching_records = {}
for header, sequence in fasta.records:
    accession = header.split('|')[1]
    if accession in filtered_accessions:
        matching_records[accession] = (header, sequence)
matching_records = list(matching_records.values())
assert len(matching_records) == len(filtered_accessions), "Mismatch between filtered accessions and matching records"

# Logging
logger.info(f'Total interactions: {len(df)}')
logger.info(f'Interactions after filtering: {len(filtered_df)}')
logger.info(f'Interactions removed: {len(df) - len(filtered_df)} ({(len(df) - len(filtered_df)) / len(df):.2%})')
logger.info(f'Total unique UniProt accessions after filtering: {len(filtered_accessions)}\n')

###############################################################################
#######                       FILTER BY LENGTH                          #######
###############################################################################

# Filter fasta records by length < 800 amino acids
short_records = []
short_accessions = set()
for header, sequence in matching_records:
    accession = header.split('|')[1]
    if len(sequence) < 800:
        short_accessions.add(accession)
        short_records.append((header, sequence))

# Logging sequences
logger.info(f'Total sequences: {len(matching_records)}')
logger.info(f'Sequences with length < 800: {len(short_records)}')
logger.info(f'Sequences with length >= 800: {len(matching_records) - len(short_records)} ({(len(matching_records) - len(short_records)) / len(matching_records):.2%})\n')

# Filter interactions by length
has_valid_length = lambda x: x in short_accessions
length_filtered_df = filtered_df[
        (filtered_df['#ID(s) interactor A'].apply(has_valid_length)) &
        (filtered_df['ID(s) interactor B'].apply(has_valid_length))
    ]

# Logging interactions
logger.info(f'Total interactions: {len(filtered_df)}')
logger.info(f'Interactions after filtering by length: {len(length_filtered_df)}')
logger.info(f'Interactions removed: {len(filtered_df) - len(length_filtered_df)} ({(len(filtered_df) - len(length_filtered_df)) / len(filtered_df):.2%})\n')

# Save filtered DataFrame
logger.info(f'Saving filtered interactions...')
length_filtered_df.to_csv(
    paths.INTACT / config.INTACT_VERSION / 'filtered.txt',
    sep='\t',
    index=False,
)

# Get sequences corresponding to the filtered interactions
filtered_accessions = set(pd.concat([
    length_filtered_df['#ID(s) interactor A'],
    length_filtered_df['ID(s) interactor B']
]).unique())
filtered_records = []
for header, sequence in short_records:
    accession = header.split('|')[1]
    if accession in filtered_accessions:
        filtered_records.append((header, sequence))

# Logging filtered sequences
logger.info(f'Sequences with length < 800 from filtered interactions: {len(filtered_records)}')
    
# Save filtered fasta file
logger.info(f'Saving filtered sequences...')
fasta = Fasta.from_records(filtered_records)
out_file = paths.INTACT / config.INTACT_VERSION / 'filtered.fasta'
fasta.write(out_file)