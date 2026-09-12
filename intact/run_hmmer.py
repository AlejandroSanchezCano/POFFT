"""
===============================================================================
Title:      Run HMMER
Outline:    If Pfam-A.hmm is not present, download it. Then, run hmmscan on the
            sequences from IntAct to detect protein families.

            We could think about parallelizing it by sequence to speed up the 
            process and perhaps go to 2 h. 
Docs:       http://eddylab.org/software/hmmer/Userguide.pdf
Author:     Alejandro Sánchez Cano
Date:       04/09/2026
Time:       14 h
===============================================================================
"""

# Built-in modules
import os
import tempfile
import subprocess

# Custom modules
from misc import paths
from fasta import Fasta
from misc.logger import logger
logger.info('Importing modules completed')

# Task ID
task = int(os.getenv('SLURM_ARRAY_TASK_ID'))
logger.info(f"Task ID: {task}")

###############################################################################
#######                    DOWNLOAD PFAM HMMER MODELS                   #######
###############################################################################

if not (paths.HMMER / 'Pfam-A.hmm').exists():
    # Download
    cmd = 'wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz'
    subprocess.run(cmd, shell=True, check=True)
    # Unzip
    cmd = 'gunzip Pfam-A.hmm.gz'
    subprocess.run(cmd, shell=True, check=True)
    # Press
    cmd = 'hmmpress Pfam-A.hmm'
    subprocess.run(cmd, shell=True, check=True)

###############################################################################
#######                           HMMSCAN                               #######
###############################################################################

# Get fasta
file_path = paths.INTACT / '2026-01-09' / 'filtered.fasta'
records = Fasta.from_file(file_path).records
record = records[task]
accession = record[0].split('|')[1]
with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:

    # Write to temporary FASTA file
    fasta = Fasta.from_records([record])
    fasta.write(temp_file.name)
    temp_file.flush()  # Ensure data is written to disk
    logger.info(f"Temporary FASTA file created: {temp_file.name}")

    # Run hmmscan
    cmd = (
        f'hmmscan'
        f' --cut_ga'
        f' --domtblout {paths.REPORTS}/hmmer/{task}.{accession}.domtblout'
        f' {paths.HMMER}/Pfam-A.hmm'
        f' {temp_file.name}'
    )
    subprocess.run(cmd, shell=True, check=True)
