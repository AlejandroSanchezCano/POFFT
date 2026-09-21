"""
===============================================================================
Title:      Fetch non-canonical sequences from UniProt
Outline:    Gather the unique valid UniProt accessions from IntAct, find the
            non-canonical ones (i.e. those with a dash in the accession), and 
            download the corresponding sequences in FASTA format. The UniProt
            REST API does not allow to fetch non-canonical sequences 
            asynchronously, so the script will fetch sequentially.
Author:     Alejandro Sánchez Cano
Date:       14/09/2026
Time:       1h 30 min
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
from fasta import Fasta
from misc.logger import logger
logger.info('Importing modules completed')

# Variables
TIMEOUT = 10  # seconds
MAX_ATTEMPTS = 5

# Gather accessions``
logger.info('Obtaining UniProt accessions...')
df = pd.read_csv(
    paths.INTACT / config.INTACT_VERSION / 'uniprot_nr.txt',
    sep='\t',
)
accessions_A = df['#ID(s) interactor A']
accessions_B = df['ID(s) interactor B']
accessions = pd.concat([accessions_A, accessions_B]).unique()
logger.info(f'Total unique UniProt accessions: {len(accessions)}')
noncanonical = [acc for acc in accessions if '-' in acc]

# Fetch
fastas = []
error_accessions = []
accession_mapping = {}
for accession in tqdm(noncanonical, desc='Fetching non-canonical sequences'):

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Perform GET request
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            break
        # Timeout
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout occurred for {accession} on attempt {attempt}/{MAX_ATTEMPTS}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
            if attempt == MAX_ATTEMPTS:
                raise Exception(f"Max attempts reached for {accession}")
        # Other exceptions
        except Exception as msg:
            response = None
            logger.error(f"Failed to fetch {accession}: {msg}\n")
            error_accessions.append(accession)
            break
    
    # Other exceptions -> break inner loop and continues with next accession
    if response is None:
        continue

    # Fill accession mapper
    fetched_accession = response.text.split('|')[1]
    accession_mapping[accession] = fetched_accession

    # A few accessions ending in '-1' return a canonical header
    if accession != fetched_accession:
        if accession != fetched_accession + '-1':
            raise ValueError(f"Unexpected mismatch: fetched {fetched_accession}, requested {accession}.")
    
    # Process the response
    fasta = Fasta.from_string(response.text)
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

# Logging
logger.info(f'Total canonical UniProt accessions: {len(accessions) - len(noncanonical)}')
logger.info(f'Total non-canonical UniProt accessions: {len(noncanonical)}')
logger.info(f'Successfully fetched sequences: {len(fasta)}')
unsuccessful = len(noncanonical) - len(fasta)
logger.info(f'Unsuccessfully fetched sequences: {unsuccessful} ({unsuccessful / len(noncanonical) * 100:.2f}%)')
logger.info(f'Unsuccessfully fetched accessions: {error_accessions}')


#if __name__ == "__main__":
#    accessions = [
#        'P08195-4', # Non-existent isoform in UniProt, but exists in IntAct
#        'P0C869-4', # Valid isoform in UniProt present in IntAct
#    ]
#    for accession in accessions:
#        print(fetch(accession))