"""
===============================================================================
Title:      Download IntAct data
Outline:    Use the IntAct class to download the data of the latest version of
            IntAct, and filter it to preserve interactions with unique UniProt
            accessions.
Docs:       https://www.ebi.ac.uk/intact/home
Author:     Alejandro Sánchez Cano
Date:       29/08/2026
Time:       10 min
===============================================================================
"""

# Third-party modules
import pandas as pd

# Custom modules
from intact import IntAct
from src.misc import paths
from src.misc.logger import logger
logger.info('Importing modules completed')

# Download IntAct data
intact = IntAct(version='2026-01-09')
#intact.download_files()

# Read the intact.txt file into a dataframe
logger.info('Reading intact.txt file...')
intact.df = pd.read_csv(
    paths.INTACT / intact.version / 'intact.txt', 
    sep='\t'
)

# Summarize database pairs
logger.info('Summarizing database pairs...')
summary = intact.summary_of_databases()
summary.to_csv(
    paths.INTACT / intact.version / 'database_count.csv',
    sep=',',
    index=False
)

# Filter accessions
intact.preserve_uniprot_accessions()
intact.remove_duplicates()
logger.info(f'Total interactions: {intact.total_interactions}')
logger.info(f'Total unique interactors: {intact.total_unique_interactors}')

# Save the processed dataframe
intact.df.to_csv(
    paths.INTACT / intact.version / 'uniprot_nr.txt',
    sep='\t',
    index=False
)