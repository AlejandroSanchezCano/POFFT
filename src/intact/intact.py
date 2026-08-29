"""
===============================================================================
Title:      IntAct
Outline:    IntAct class to download the data of any version of IntAct, reduce 
            it to only plant interactors, search for all MADS interactions, and 
            filter to only MADS vs. MADS interactions. They are represented by 
            their UniProt accessions.
Docs:       https://www.ebi.ac.uk/intact/home
Author:     Alejandro Sánchez Cano
Date:       17/10/2025
===============================================================================
"""

# Built-in modules
import re
import subprocess
from io import StringIO

# Third-party modules
import pandas as pd

# Custom modules
from src.misc import paths
from src.misc.logger import logger

class IntAct:

    def __init__(self, version: str):
        self.version = version

    def download_files(self) -> None:
        '''
        Downloads 'intact.txt' file from the specified IntAct version.
        '''

        # Set up output directory
        output_dir = paths.INTACT / self.version
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download file
        url_file = f'https://ftp.ebi.ac.uk/pub/databases/intact/{self.version}/psimitab/intact.zip'
        wget = f'wget {url_file} -P {output_dir} -q'
        subprocess.run(wget, shell = True)

        # Unzip file and remove compressed and negatives file
        unzip = f'unzip -qq {output_dir}/intact.zip -d {output_dir}'
        subprocess.run(unzip, shell = True)
        rm = f'rm {output_dir}/intact.zip'
        subprocess.run(rm, shell = True)
        rm = f'rm {output_dir}/intact_negative.txt'
        subprocess.run(rm, shell = True)

        # Logging
        logger.info(f'IntAct {self.version} "intact.txt" file downloaded')