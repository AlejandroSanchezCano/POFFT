"""
===============================================================================
Title:      Download IntAct data
Outline:    Use the IntAct class to download the data of the latest version of
            IntAct, and filter it to preserve interactions with unique UniProt
            accessions.
Docs:       https://www.ebi.ac.uk/intact/home
Author:     Alejandro Sánchez Cano
Date:       29/08/2026
Time:       5 min
===============================================================================
"""

# Third-party modules
from intact import IntAct

# Custom modules
from src.misc import paths
from src.misc.logger import logger

# Download and process the IntAct data
intact = IntAct(version='2025-08-08')
intact.download_files()
intact.preserve_uniprot_accessions()
intact.remove_duplicates()
logger.info(f'Total interactions: {intact.total_interactions}')
logger.info(f'Total unique interactors: {intact.total_unique_interactors}')

# Save the processed dataframe
intact.df.to_csv(
    paths.INTACT / intact.version / 'filtered.txt',
    sep='\t',
    index=False
)