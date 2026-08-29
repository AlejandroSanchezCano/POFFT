"""
===============================================================================
Title:      IntAct
Outline:    IntAct class to download the data of any version of IntAct, and 
            filter it to preserve only UniProt accessions, and remove duplicate
            interactions.
Docs:       https://www.ebi.ac.uk/intact/home
Author:     Alejandro Sánchez Cano
Date:       29/08/2026
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
        self.df = None

    @property
    def total_interactions(self) -> int:
        '''
        Total number of interactions as represented by the number of lines in
        the dataframe.

        Returns
        -------
        int
            Total number of interactions.
        '''
        if self.df is None:
            raise ValueError('self.df attribute is None, load the IntAct data first')
        return len(self.df)
    
    @property
    def total_unique_interactors(self) -> int:
        '''
        Total number of unique interactors as represented by the number of
        unique accessions in the columns "#ID(s) interactor A" and
        "ID(s) interactor B".

        Returns
        -------
        int
            Total number of unique interactors.
        '''
        if self.df is None:
            raise ValueError('self.df attribute is None, load the IntAct data first')
        col1 = '#ID(s) interactor A'
        col2 = 'ID(s) interactor B'
        return len(set(self.df[col1]).union(set(self.df[col2])))

    def download_files(self) -> None:
        '''
        Downloads 'intact.txt' file from the specified IntAct version.
        '''
        # Set up output directory
        output_dir = paths.INTACT / self.version
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download file
        logger.info(f'Downloading IntAct {self.version} "intact.zip" file...')
        url_file = f'https://ftp.ebi.ac.uk/pub/databases/intact/{self.version}/psimitab/intact.zip'
        wget = f'wget {url_file} -P {output_dir} -q'
        subprocess.run(wget, shell = True)

        # Unzip file and remove compressed file
        unzip = f'unzip -qq {output_dir}/intact.zip -d {output_dir}'
        subprocess.run(unzip, shell = True)
        rm = f'rm {output_dir}/intact.zip'
        subprocess.run(rm, shell = True)

        # Logging
        logger.info(f'IntAct {self.version} files downloaded')

    def preserve_uniprot_accessions(self) -> None:
        '''
        The columns "ID(s) interactor A" and "ID(s) interactor B" contain 
        accessions from different databases such as UniProt, IntAct or even 
        ChEBI. UniProt accessions are the vast majority and the ones from which
        the protein sequence can be retrieved easily. Thus, this method removes
        all lines with accessions from other databases than UniProt. It also 
        removes lines with UniProt accessions containing the string "PRO", 
        which are protein fragments and not full-length proteins.

        It would be slighly faster with awk or other command line tools, but
        using pandas gives more flexibility, plus the fie is not that big and
        the method is only called once per version.
        '''
        # Convert to dataframe
        logger.info('Reading intact.txt file...')
        df = pd.read_csv(
            paths.INTACT / self.version / 'intact.txt', 
            sep='\t'
        )
        total_lines = len(df)

        # Distribution of databases
        which_accession = lambda x: x.split(':')[0]
        col1 = '#ID(s) interactor A'
        col2 = 'ID(s) interactor B'
        counts = df[[col1, col2]].map(which_accession).value_counts()
        logger.debug(f'Accessions counts:\n{counts}')

        # Filter out non-UniProt accessions
        logger.info(f'Filtering out non-UniProt accessions...')
        df = df[
                (df[col1].apply(which_accession) == 'uniprotkb') &
                (df[col2].apply(which_accession) == 'uniprotkb')
            ]
        non_uniprot_lines = total_lines - len(df)
        logger.info(f'Total lines: {total_lines}')
        logger.info(f'Lines after filtering: {len(df)}')
        logger.info(f'Lines removed: {non_uniprot_lines} ({non_uniprot_lines/total_lines:.2%})')

        # Filter out UniProt accessions containing "PRO"
        logger.info(f'Filtering out UniProt accessions containing "PRO"...')
        contains_PRO = lambda x: "PRO" in x
        df = df[
                ~df[col1].apply(contains_PRO) &
                ~df[col2].apply(contains_PRO)
            ]
        pro_lines = total_lines - non_uniprot_lines - len(df)
        logger.info(f'Total lines: {total_lines - non_uniprot_lines}')
        logger.info(f'Lines after filtering: {len(df)}')
        logger.info(f'Lines removed: {pro_lines} ({pro_lines/(total_lines - non_uniprot_lines):.2%})')

        # Save filtered dataframe
        self.df = df

    def remove_duplicates(self) -> None:
        '''
        IntAct encodes interactions in a single line, but if the same 
        interaction is found from different experimental methods, each method
        will be represented in a different line. This method removes duplicate
        interaction lines, regardless of the interactor order (AB = BA).
        '''
        # Create a new column with sorted 
        logger.info(f'Removing duplicate interactions...')
        col1 = '#ID(s) interactor A'
        col2 = 'ID(s) interactor B'
        self.df['sorted_accessions'] = self.df[[col1, col2]].apply(
            lambda x: tuple(sorted(x)), axis=1
        )

        # Remove duplicates based on the new column
        total_lines = len(self.df)
        self.df.drop_duplicates(subset='sorted_accessions', inplace=True)
        duplicates_removed = total_lines - len(self.df)
        logger.info(f'Total lines: {total_lines}')
        logger.info(f'Lines after filtering: {len(self.df)}')
        logger.info(f'Lines removed: {duplicates_removed} ({duplicates_removed/total_lines:.2%})')

        # Remove temp column
        self.df.drop(columns='sorted_accessions', inplace=True)