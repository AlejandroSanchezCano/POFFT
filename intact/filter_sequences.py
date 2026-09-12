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
import sys

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from fasta import Fasta
from misc import paths
from misc.logger import logger
logger.info('Importing modules completed')

###############################################################################
#######                   VERIFY NO MISSING ACCESSION                   #######
###############################################################################

# Gather total accessions
df = pd.read_csv(
    paths.INTACT / '2026-01-09' / 'uniprot_nr.txt',
    sep='\t',
)
total_interactions = len(df)
accessions_A = df['#ID(s) interactor A'].apply(lambda x: x.split(':')[1])
accessions_B = df['ID(s) interactor B'].apply(lambda x: x.split(':')[1])
total_accessions = pd.concat([accessions_A, accessions_B]).unique()
logger.info(f'Total UniProt accessions: {len(total_accessions)}')

# Gather fetched accessions
fetched_accessions = os.listdir(paths.REPORTS / 'uniprotjob')
fetched_accessions = [f.split('.')[0] for f in fetched_accessions]
logger.info(f'Fetched UniProt accessions: {len(fetched_accessions)}')

# Gather failed accessions
failed_accessions = [
    'A4GZ26', 
    'P08195-4', 
    'P0C869-4', 
    'P13706-1', 
    'P61966-2', 
    'P63059-1', 
    'P85299-2', 
    'Q13421-2', 
    'Q86VZ6-2', 
    'Q8CGF4'
] 
logger.info(f'Failed UniProt accessions: {len(failed_accessions)}')

# Verify that all accessions have been fetched
missing_accessions = set(total_accessions) - set(fetched_accessions) - set(failed_accessions)
if missing_accessions:
    logger.error(f"Missing accessions: {missing_accessions}. Rerun the fetch_sequences.py script to retrieve them.")
    sys.exit(1)
else:
    logger.info("All accessions have been fetched or failed, no missing accessions\n")

###############################################################################
#######              DISREGARD ACCESSIONS WITHOUT SEQUENCE              #######
###############################################################################

# UniProt accessions from headers
fasta_path = paths.INTACT / '2026-01-09' / 'sequences.fasta'
fasta = Fasta.from_file(fasta_path)
fasta_accessions = [header.split('|')[1] for header in fasta.headers]
logger.info(f'Total UniProt accessions in fasta file: {len(fasta_accessions)}')
fasta_accessions = set(fasta_accessions)
logger.info(f'Unique UniProt accessions in fasta file: {len(fasta_accessions)}')

# Remove rows without fetched sequences
has_fetched_sequence = lambda x: x.split(':')[1] in fasta_accessions
col1 = '#ID(s) interactor A'
col2 = 'ID(s) interactor B'
df = df[
        (df[col1].apply(has_fetched_sequence)) &
        (df[col2].apply(has_fetched_sequence))
    ]
accessions_A = df['#ID(s) interactor A'].apply(lambda x: x.split(':')[1])
accessions_B = df['ID(s) interactor B'].apply(lambda x: x.split(':')[1])
filtered_accessions = set(pd.concat([accessions_A, accessions_B]).unique())

# Filter fasta records to only include those matching in the filtered DataFrame
matching_records = {}
for header, sequence in fasta.records:
    accession = header.split('|')[1]
    if accession in filtered_accessions:
        matching_records[accession] = (header, sequence)
matching_records = list(matching_records.values())
assert len(matching_records) == len(filtered_accessions), "Mismatch between filtered accessions and matching records"

# Logging
logger.info(f'Total interactions: {total_interactions}')
logger.info(f'Interactions after filtering: {len(df)}')
logger.info(f'Interactions removed: {total_interactions - len(df)} ({(total_interactions - len(df)) / total_interactions:.2%})')
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
logger.info(f'Saving filtered interactions to {paths.INTACT / "2026-01-09" / "filtered.txt"}...')
length_filtered.to_csv(
    paths.INTACT / '2026-01-09' / 'filtered.txt',
    sep='\t',
    index=False,
)

# Get sequences corresponding to the filtered interactions
accessions_A = length_filtered['#ID(s) interactor A'].apply(lambda x: x.split(':')[1])
accessions_B = length_filtered['ID(s) interactor B'].apply(lambda x: x.split(':')[1])
filtered_accessions = set(pd.concat([accessions_A, accessions_B]).unique())
filtered_records = []
for header, sequence in short_records:
    accession = header.split('|')[1]
    if accession in filtered_accessions:
        filtered_records.append((header, sequence))

# Logging filtered sequences
logger.info(f'Sequences with length < 800 from filtered interactions: {len(filtered_records)}')
    
# Save filtered fasta file
logger.info(f'Saving filtered sequences to {paths.INTACT / "2026-01-09" / "filtered.fasta"}...')
fasta = Fasta.from_records(filtered_records)
out_file = paths.INTACT / '2026-01-09' / 'filtered.fasta'
fasta.write(out_file)