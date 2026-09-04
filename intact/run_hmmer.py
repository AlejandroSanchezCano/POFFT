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
import subprocess

# Custom modules
from misc import paths

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

# Run hmmscan
cmd = (
    f'hmmscan'
    f' --cut_ga'
    f' --domtblout {paths.INTACT}/2026-01-09/hmmer.domtblout'
    f' {paths.HMMER}/Pfam-A.hmm'
    f' {paths.INTACT}/2026-01-09/filtered.fasta'
)
subprocess.run(cmd, shell=True, check=True)