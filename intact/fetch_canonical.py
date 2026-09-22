"""
===============================================================================
Title:      Fetch sequences from UniProt
Outline:    Gather the unique valid UniProt accessions from IntAct, batch them,
            and use the UniProt REST API wrapped in the UniProtJob class to
            fetch the corresponding sequences in FASTA format.
            The UniProt REST API is finnicky, and although retrying is
            implemented, it might unexpectedly fail, in which case just re-run
            the script and it will eventually work. 
            After a substantial number of requests, the UniProt REST API might
            decide to block the IP address for a while, in which case try later
            with increased poll interval.
Author:     Alejandro Sánchez Cano
Date:       02/09/2026
Time:       50 min
===============================================================================
"""

# Built-in modules
import os
import json
import time
import random

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from misc import paths
from misc import config
from tools.fasta import Fasta
from misc.logger import logger
from uniprotjob import UniProtJob
logger.info('Importing modules completed')

# Variables
BATCH_SIZE = 10_000
PERMUTE = True
MAX_ATTEMPTS = 5
POLL_INTERVAL = 10  # seconds
MAX_WAIT_TIME = 5 # min
PAGINATION_SIZE = 400

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
accessions = [acc for acc in accessions if '-' not in acc]
logger.info(f'Total canonical accessions: {len(accessions)}')

# Accession mapping
logger.info('Loading accession mapping json...')
mapper_path = paths.INTACT / config.INTACT_VERSION / 'accession_mapping.json'
with open(mapper_path, 'r') as handle:
    accession_mapping = json.load(handle)

# Manually map problematic accessions
manual_mapping = {
    'A4GZ26': 'Q5DU25',
    'Q8CGF4': 'G3X972',
}
accessions = [manual_mapping.get(acc, acc) for acc in accessions]
accession_mapping.update(manual_mapping)

# Check for already fetched accessions
fetched = os.listdir(paths.REPORTS / 'uniprotjob')
fetched = [f.split('.')[0] for f in fetched]
remaining_accessions = list(set(accessions) - set(fetched))
if len(remaining_accessions) != len(accessions):
    logger.info(f'Accessions already fetched: {len(fetched)}')
    logger.info(f'Remaining accessions to fetch: {len(remaining_accessions)}')
    accessions = remaining_accessions

# Batch accessions
batches = [
    accessions[idx:idx + BATCH_SIZE]
    for idx in range(0, len(accessions), BATCH_SIZE)
]

# Process batches
fasta_records = []
for idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
    logger.info(f"Batch {idx + 1}/{len(batches)} with {len(batch)} accessions")

    # Permute accessions in the batch to avoid potential API issues
    if PERMUTE:
        random.shuffle(batch)

    # Retry loop
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Process batch
        try:
            job = UniProtJob(batch)
            job.submit()
            job.wait(
                poll_interval=POLL_INTERVAL, 
                max_wait_time=MAX_WAIT_TIME*60
            )
            break
        # Handle exceptions
        except Exception as e:
            logger.error(
                f"Error processing batch {idx + 1} on attempt {attempt}: "
                f"{e}\n"
            )
            if attempt == MAX_ATTEMPTS:
                raise
        # Wait before retrying
        waiting_time = 2 ** attempt * 5
        logger.info(f"Retrying in {waiting_time}s...\n")
        time.sleep(waiting_time)

    # Download sequences and accession mapping
    sequences, mapping = job.download(
        size=PAGINATION_SIZE,
        json_dir=paths.REPORTS / 'uniprotjob'
    )
    fasta = Fasta.from_string(sequences)
    fasta_records.extend(fasta.records)
    accession_mapping.update(mapping)
    logger.info(f"Downloaded {len(fasta)} sequences from batch {idx + 1}")

# Save FASTA records to file
out_path = paths.INTACT / config.INTACT_VERSION / 'sequences.fasta'
fasta = Fasta.from_records(fasta_records)
fasta.write(out_path=out_path, mode='a')
logger.info(f"{len(fasta)} sequences saved to {out_path}")

# Save accession mapping to file
with open(mapper_path, 'w') as handle:
    json.dump(accession_mapping, handle, indent=4)
