"""
===============================================================================
Title:      Logger module
Outline:    Manage logging for the project. It includes a custom logger that
            logs to the standard output, and another one that logs to a file
            in the logs directory, which has the same folder structure as the
            src directory.
Docs:       https://docs.python.org/3/library/logging.html
Author:     Alejandro Sánchez Cano
Date:       28/08/2026
===============================================================================
"""

# Built-in modules
import logging
import inspect
from pathlib import Path

# Custom modules
from misc import paths

# Standard output handler
def build_stdout_handler() -> logging.StreamHandler:
    '''
    Each file logs to standard output

    Returns
    -------
    logging.StreamHandler
        Standard output handler
    '''
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(filename)s %(levelname)s %(message)s')
    stdout_handler.setFormatter(formatter)
    return stdout_handler

# File handler
def build_file_handler() -> logging.FileHandler:
    '''
    Each file logs to the logs directory

    Returns
    -------
    logging.FileHandler
        File handler
    '''
    main_file = inspect.stack()[-1].filename
    main_file = Path(main_file)
    logs_file = paths.PROJECT / 'logs' / main_file.relative_to(paths.PROJECT)
    logs_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        logs_file.with_suffix('.log'),
        mode='w'
        )
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(levelname)s %(message)s', 
        datefmt='%d/%m/%Y %H:%M:%S')
    file_handler.setFormatter(formatter)
    return file_handler

# Custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(build_stdout_handler())
if 'tests' not in inspect.stack()[-1].filename:
    logger.addHandler(build_file_handler())