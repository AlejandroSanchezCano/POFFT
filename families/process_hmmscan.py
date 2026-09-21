"""
===============================================================================
Title:      Process hmmscan results
Outline:    After running hmmscan on the sequences from IntAct, process the 
            results by merging the several domtblout files (from array jobs)
            into a single file. Then, filter by i_evalue threshold and keep 
            the significant hits. Finally, filter the interactions and
            sequences to keep only those with hmmscan hits. Save the resulting
            files for further analysis.
Author:     Alejandro Sánchez Cano
Date:       17/09/2026
Time:       2 min
===============================================================================
"""

# Third-party modules
import pandas as pd

# Custom modules
from misc import paths
from misc import config
from fasta import Fasta
from misc.logger import logger
from domtblout import Domtblout
logger.info('Importing modules completed')

###############################################################################
#######                     MERGE DOMTBLOUT FILES                       #######
###############################################################################

# Combine domtblout files
domtblout_dir = paths.REPORTS / 'hmmer'
domtblout_files = sorted(
    domtblout_dir.iterdir(), 
    key=lambda x: int(x.name.split('.')[0])
)
df = pd.concat(
    [Domtblout.from_file(file).df for file in domtblout_files],
    ignore_index=True
)
domtblout = Domtblout.from_df(df)
logger.info(
    f'Combined {len(domtblout_files)} domtblout files into a single dataframe '
    f'with {len(df)} Pfam hits and {len(df["query_name"].unique())} '
    f'unique queries\n'
)

# Save combined dataframe
df.to_csv(
    paths.FAMILIES / 'hmmscan.domtblout', 
    index=False,
    sep='\t'
)

###############################################################################
#######                 KEEP ONLY THOSE WITH HMMSCAN HITS               #######
###############################################################################

# Filter by i_evalue threshold
domtblout.evalue_filter(
    field='i_evalue', 
    threshold=config.IEVALUE_THRESHOLD
)
logger.info(
    f'Filtered Pfam hits by i_evalue threshold of {config.IEVALUE_THRESHOLD}, '
    f'keeping {len(domtblout.df)} hits and '
    f'{len(domtblout.df["query_name"].unique())} unique queries'
)

# Load interactions
logger.info(f'Loading interactions file...')
interactions = pd.read_csv(
    paths.INTACT / config.INTACT_VERSION / 'filtered.txt',
    sep='\t',
)
logger.info(f'Total interactions: {len(interactions)}')

# Filter interactions by accessions with hmmscan hits
queries = domtblout.df['query_name']
accessions = queries.apply(lambda x: x.split('|')[1])
accessions = set(accessions.unique())
with_hits = interactions[
    (interactions['#ID(s) interactor A'].isin(accessions)) &
    (interactions['ID(s) interactor B'].isin(accessions))
]
logger.info(
    f'Interactions of both proteins with hmmscan hits: '
    f'{len(with_hits)}'
)
logger.info(
    f'Interactions removed: {len(interactions) - len(with_hits)} '
    f'({(len(interactions) - len(with_hits)) / len(interactions):.2%})\n'
)

# Save interaction file
logger.info(f'Saving filtered interactions...')
with_hits.to_csv(
    paths.FAMILIES / 'interactions.txt',
    sep='\t',
    index=False,
)

# Load fasta file
fasta_path = paths.INTACT / config.INTACT_VERSION / 'filtered.fasta'
fasta = Fasta.from_file(fasta_path)

# Filter fasta records by accessions with hmmscan hits
matching_records = []
for header, sequence in fasta.records:
    accession = header.split('|')[1]
    if accession in accessions:
        matching_records.append((header, sequence))
assert len(matching_records) == len(accessions), (
    f'Number of matching records ({len(matching_records)}) does not match '
    f'number of accessions ({len(accessions)})'
)
logger.info(f'Total sequences: {len(fasta)}')
logger.info(f'Sequences with hmmscan hits: {len(matching_records)}')
logger.info(
    f'Sequences removed: {len(fasta) - len(matching_records)} '
    f'({(len(fasta) - len(matching_records)) / len(fasta):.2%})\n'
)  

# Save filtered fasta file
logger.info(f'Saving filtered sequences...')
fasta = Fasta.from_records(matching_records)
fasta.write(paths.FAMILIES / 'sequences.fasta')