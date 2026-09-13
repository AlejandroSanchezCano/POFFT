"""
===============================================================================
Title:      Run hmmscan
Outline:    I Pfam-A.hmm is not present, download it, although this should be 
            done manually. Then, run hmmscan on the sequences from IntAct to
            detect protein families.
Docs:       http://eddylab.org/software/hmmer/Userguide.pdf
Author:     Alejandro Sánchez Cano
Date:       04/09/2026
Time:       14 h sequential,  40 min with 20 array jobs
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
TASK = int(os.getenv('SLURM_ARRAY_TASK_ID'))
TOTAL_TASKS = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
logger.info(f"Task ID: {TASK}, Total Tasks: {TOTAL_TASKS}")

###############################################################################
#######                    DOWNLOAD PFAM HMMER MODELS                   #######
###############################################################################

if not (paths.HMMER / 'Pfam-A.hmm').exists():
    # cd 
    cmd = f'cd {paths.HMMER}'
    subprocess.run(cmd, shell=True, check=True)
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
records = Fasta.from_file(file_path).records[TASK::TOTAL_TASKS]
logger.info(f"Task {TASK}: Processing {len(records)} records from {file_path}")
with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp_file:

    # Write to temporary FASTA file
    fasta = Fasta.from_records(records)
    fasta.write(temp_file.name)
    temp_file.flush()  # Ensure data is written to disk
    logger.info(f"Temporary FASTA file created: {temp_file.name}")

    # Run hmmscan
    cmd = (
        f'hmmscan'
        f' --cut_ga'
        f' --domtblout {paths.REPORTS}/hmmer/{TASK}.domtblout'
        f' {paths.HMMER}/Pfam-A.hmm'
        f' {temp_file.name}'
    )
    subprocess.run(cmd, shell=True, check=True)
