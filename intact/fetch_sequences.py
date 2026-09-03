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
Time:       20 min
===============================================================================
"""

# Built-in modules
import time

# Third-party modules
import pandas as pd
from tqdm import tqdm

# Custom modules
from misc import paths
from fasta import Fasta
from misc.logger import logger
from uniprotjob import UniProtJob
logger.info('Importing modules completed')

# Gather accessions
logger.info('Obtaining UniProt accessions...')
df = pd.read_csv(
    paths.INTACT / '2026-01-09' / 'uniprot_nr.txt',
    sep='\t',
)
accessions_A = df['#ID(s) interactor A'].apply(lambda x: x.split(':')[1])
accessions_B = df['ID(s) interactor B'].apply(lambda x: x.split(':')[1])
accessions = pd.concat([accessions_A, accessions_B]).unique()
logger.info(f'Total unique UniProt accessions: {len(accessions)}')

# Batch accessions
BATCH_SIZE = 10_000
batches = [
    accessions[idx:idx + BATCH_SIZE]
    for idx in range(0, len(accessions), BATCH_SIZE)
]

# Process batches
records = []
MAX_ATTEMPTS = 5
for idx, batch in enumerate(tqdm(batches, desc="Processing batches")):

    logger.info(f"Batch {idx + 1}/{len(batches)} with {len(batch)} accessions")

    # Retry loop
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Process batch
        try:
            job = UniProtJob(batch)
            job.submit()
            job.wait(poll_interval=10, max_wait_time=5*60)
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

    # Downoad sequences
    response = job.download(size=400)
    fasta = Fasta.from_string(response)
    records.extend(fasta.records)
    logger.info(f"Downloaded {len(fasta)} sequences from batch {idx + 1}")

# Save results
out_path = paths.INTACT / '2026-01-09' / 'sequences.fasta'
fasta = Fasta.from_records(records)
fasta.write(out_path=out_path)
logger.info(f"{len(fasta)} sequences saved to {out_path}")