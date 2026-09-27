"""
===============================================================================
Title:      Fetch sequences from UniProt
Outline:    Many strategies exist to fetch sequences from UniProt, most of them
            involving the use of the UniProt REST API. This script creates a
            custom per-accession endpoint and sends a GET request to fetch the
            corresponding sequence in FASTA format. The fetched sequences are
            then saved to a FASTA file, and a mapping of original to fetched
            accessions is saved to a JSON file. This strategy is slow but 
            robust.
Author:     Alejandro Sánchez Cano
Date:       27/09/2026
Time:       8 h
===============================================================================
"""

# Built-in modules
import time
import json
import requests

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from misc import paths
from misc import config
from tool.fasta import Fasta
from misc.logger import logger
logger.info('Importing modules completed')

# Variables
TIMEOUT = 10  # seconds
MAX_ATTEMPTS = 5

###############################################################################
#######                       GATHER ACCESSIONS                         #######
###############################################################################

# Gather accessions
logger.info('Obtaining UniProt accessions...')
df = pd.read_csv(
    paths.INTACT / config.INTACT_VERSION / 'uniprot_nr.txt',
    sep='\t',
)
accessions_A = df['#ID(s) interactor A']
accessions_B = df['ID(s) interactor B']
accessions = pd.concat([accessions_A, accessions_B]).unique()
logger.info(f'Total unique UniProt accessions: {len(accessions)}')

###############################################################################
#######                    FETCH SEQUENCE FUNCTION                      #######
###############################################################################

def fetch(accession: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Perform GET request
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text.strip()
        # Timeout error
        except requests.exceptions.Timeout:
            logger.warning(
                f'Timeout occurred for {accession} on attempt '
                f'{attempt}/{MAX_ATTEMPTS}. Retrying...'
            )
            time.sleep(2 ** attempt) # Exponential backoff
            continue
        # Connection error
        except requests.exceptions.ConnectionError:
            logger.warning(
                f'Connection error occurred for {accession} on attempt '
                f'{attempt}/{MAX_ATTEMPTS}. Retrying...'
            )
            time.sleep(2 ** attempt) # Exponential backoff
            continue
        # HTTP error
        except requests.HTTPError as e:
            logger.error(f'Failed to fetch {accession}: {e}')
            return None
        # Other exceptions
        except Exception as e:
            logger.error(f'Unexpected error occurred for {accession}: {e}')
            raise e

    # Maximum attempts reached
    raise Exception(f"Max attempts reached for {accession}")
    
###############################################################################
#######                    LOOP THROUGH ACCESSIONS                      #######
###############################################################################

# Fetching loop
fastas = []
accession_mapping = {}
with requests.Session() as session:
    for accession in tqdm(accessions, desc='Fetching sequences'):

        # Fetch the sequence
        text = fetch(accession)
        if text is None: continue

        # Empty response
        if text == '':
            logger.error(f'Empty response for {accession}')
            continue

        # Fill accession mapper
        fetched_accession = text.split('|')[1]
        accession_mapping[accession] = fetched_accession

        # Process the response
        fasta = Fasta.from_string(text)
        fastas.append(fasta)

# Save the fetched sequences to a FASTA file
logger.info('Saving fetched sequences to FASTA file...')
out_path = paths.INTACT / config.INTACT_VERSION / 'sequences.fasta'
fasta = Fasta.concatenate(fastas)
fasta.write(out_path)

# Save accession mapper
mapper_path = paths.INTACT / config.INTACT_VERSION / 'accession_mapping.json'
with open(mapper_path, 'w') as handle:
    json.dump(accession_mapping, handle, indent=4)
