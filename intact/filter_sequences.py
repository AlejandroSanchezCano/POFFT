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

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from fasta import Fasta
from misc import paths
from misc.logger import logger
logger.info('Importing modules completed')

###############################################################################
#######              DISREGARD ACCESSIONS WITHOUT SEQUENCE              #######
###############################################################################

# UniProt accessions from headers
fasta_path = paths.INTACT / '2026-01-09' / 'sequences.fasta'
fasta = Fasta.from_file(fasta_path)
accessions = [header.split('|')[1] for header in fasta.headers]
logger.info(f'Total UniProt accessions: {len(accessions)}')
accessions = set(accessions)
logger.info(f'Unique UniProt accessions: {len(accessions)}')

# Load filtered DataFrame
df = pd.read_csv(
    paths.INTACT / '2026-01-09' / 'uniprot_nr.txt',
    sep='\t',
)
total_interactions = len(df)

# Remove rows without fetched sequences
has_sequence = lambda x: x.split(':')[1] in accessions
col1 = '#ID(s) interactor A'
col2 = 'ID(s) interactor B'
df = df[
        (df[col1].apply(has_sequence)) &
        (df[col2].apply(has_sequence))
    ]

# Logging
logger.info(f'Total interactions: {total_interactions}')
logger.info(f'Interactions after filtering: {len(df)}')
logger.info(f'Interactions removed: {total_interactions - len(df)} ({(total_interactions - len(df)) / total_interactions:.2%})')
accessions_A = df['#ID(s) interactor A'].apply(lambda x: x.split(':')[1])
accessions_B = df['ID(s) interactor B'].apply(lambda x: x.split(':')[1])
accessions = pd.concat([accessions_A, accessions_B]).unique()
logger.info(f'Total unique UniProt accessions after filtering: {len(accessions)}')

###############################################################################
#######                       FILTER BY LENGTH                          #######
###############################################################################

# Filter UniProt accessions
short_accessions = set()
filtered_records = []
sequence_length_threshold = 800
for header, sequence in tqdm(fasta.records, desc='Filtering by length'):
    accession = header.split('|')[1].split('|')[0]
    if len(sequence) < sequence_length_threshold:
        short_accessions.add(accession)
        filtered_records.append((header, sequence))

# Logging interactors
logger.info(f'Total sequences: {len(accessions)}')
logger.info(f'Sequences with length < {sequence_length_threshold}: {len(short_accessions)}')
logger.info(f'Sequences with length >= {sequence_length_threshold}: {len(accessions) - len(short_accessions)} ({(len(accessions) - len(short_accessions)) / len(accessions):.2%})\n')

# Filter interactions by length
has_valid_length = lambda x: x.split(':')[1] in short_accessions
length_filtered = df[
        (df[col1].apply(has_valid_length)) &
        (df[col2].apply(has_valid_length))
    ]

# Logging interactions
logger.info(f'Total interactions: {len(df)}')
logger.info(f'Interactions after filtering by length: {len(length_filtered)}')
logger.info(f'Interactions removed: {len(df) - len(length_filtered)} ({(len(df) - len(length_filtered)) / len(df):.2%})\n')

# Save filtered DataFrame
length_filtered.to_csv(
    paths.INTACT / '2026-01-09' / 'filtered.txt',
    sep='\t',
    index=False,
)

# Save filtered fasta file
fasta = Fasta.from_records(filtered_records)
out_file = paths.INTACT / '2026-01-09' / 'filtered.fasta'
fasta.write(out_file)