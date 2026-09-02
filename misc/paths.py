"""
===============================================================================
Title:      Paths module
Outline:    Important path variables for the project to avoid hardcoding them
            in each individual file.
Author:     Alejandro Sánchez Cano
Date:       28/08/2026
===============================================================================
"""

# Built-in modules
from pathlib import Path

# Top directories
CHONKY = Path('/gpfs/work5/0/prjs1460/')
PROJECT = CHONKY / 'POFFT'
TOOLS = CHONKY / 'tools'

# Data directories
DATA = PROJECT / 'data'
INTACT = DATA / 'intact'