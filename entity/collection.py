"""
===============================================================================
Title:      Collection
Outline:    Collection class that represents a collection of entities such as
            Proteins, PPIs, etc. It hosts functionalities that concern 
            collections as a whole and not individual entities.
            + ProteinCollection:
              - Load and save Protein objects from/to HDF5 files.
              - Iterate over Protein objects in the collection.
              - Generate sequence reports (length distributions, etc).
              - Export sequences to FASTA files (single or per species).
            + PPICollection:
              - Load and save PPI objects from/to HDF5 files.
              - Iterate over PPI objects in the collection.
Author:     Alejandro Sánchez Cano
Date:       17/09/2026
===============================================================================
"""

# Built-in modules
import json
from pathlib import Path
from typing import Iterable
from abc import ABC, abstractmethod
from collections import defaultdict

# Third-party modules
import h5py
import numpy as np
from tqdm import tqdm

# Custom modules
from misc import paths
from .protein import Protein
from .pair import ProteinPair
from misc.logger import logger

class Collection(ABC):

###############################################################################
#########                     ABSTRACT METHODS                        #########
###############################################################################

    @abstractmethod
    def __init__(
        self, 
        file_path: str | Path | None = None, 
        items: list | None = None
        ):
        pass

    @abstractmethod
    def __iter__(self):
        pass 
    
    def __contains__(self, item) -> bool:
        return any(x == item for x in self)

    @abstractmethod
    def to_hdf5(self) -> None:
        pass

###############################################################################
#########                       CLASS UTILS                           #########
###############################################################################

    def hierarchy(
        self, 
        h5object: h5py.File | h5py.Group | None = None,
        indent: int = -1
        ) -> None:
        # Open HDF5 file
        if h5object is None:
            with h5py.File(self.file_path, "r") as file:
                self.hierarchy(h5object=file, indent=indent + 1)
                return

        # Inspect the current HDF5 object
        for key, item in h5object.items():
            print("\t" * indent + key, end='')
            if isinstance(item, h5py.Dataset):
                print(f" (Dataset, shape={item.shape}, dtype={item.dtype})")
            elif isinstance(item, h5py.Group):
                print(" (Group)")
                self.hierarchy(item, indent + 1)

    def _pathify(
        self,
        lst: list,
        parent: str = '',
        ) -> list:
        """
        Converts a list of strings and nested dictionaries into a dictionary 
        mapping each path to a boolean indicating whether it's a leaf node.
        E.g. ['a', 'b', {'c': ['d', 'e']}] -> {'a': True, 'b': True, 
        'c': False, 'c/d': True, 'c/e': True}
        """
        paths = {}
        for item in lst:
            if not isinstance(item, dict):
                append = f"{parent}/{item}" if parent else item
                paths[append] = True
            else:
                for key, value in item.items():
                    append = f"{parent}/{key}" if parent else key
                    paths[append] = False
                    paths.update(self._pathify(value, parent=append))
        return paths

    def _should_load(
        self, 
        dataset: str
        ) -> bool:
        '''
        Checks if a dataset should be loaded based on the collection's 
        configuration. Loading is differential, meaning that only the datasets 
        specified in self.datasets are loaded.

        Parameters
        ----------
        dataset : str
            Name of the dataset to check.
        Returns
        -------
        bool
            Whether the dataset should be loaded.
        '''
        
        # All datasets
        if self.datasets is None:
            return True
        
        # Lightweight
        if self.datasets == 'lightweight':
            return dataset in self.LIGHTWEIGHT

        # Nested datasets
        if '/' in dataset:
            levels = dataset.split('/')
            for idx in range(1, len(levels) + 1):
                partial = '/'.join(levels[:idx])
                if partial not in self.datasets: 
                    return False
                if self.datasets[partial]:
                    return True

            return True

        # Specific datasets
        if isinstance(self.datasets, dict):
            return dataset in self.datasets
        
        return False

    def _is_default(self, value) -> bool:
        '''
        Checks if a value is the default value (None, empty dict, or empty 
        list, ...).

        Parameters
        ----------
        value : any
            Value to check.
        
        Returns
        -------
        bool
            Whether the value is the default value.
        '''
        if value is None:
            return True
        if isinstance(value, dict) and len(value) == 0:
            return True
        if isinstance(value, list) and len(value) == 0:
            return True
        if isinstance(value, np.ndarray):
            return value.size == 0 or np.all(np.isnan(value))
        if isinstance(value, np.floating):  # e.g. np.float32
            return np.isnan(value)
        return False

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ProteinCollection(Collection):

    DATASETS_DTYPES = {
        "seq": "str",
        "uniprot": "str",
        "taxon": "int",
        "family": "json",
        "esm2_embeddings": "group",
    }

    ESM_MODELS_DIMS = {
        "8M": 320,
        "35M": 480,
        "150M": 640,
        "650M": 1280,
        "3B": 2560,
        "15B": 5120,
    }

    LIGHTWEIGHT = [
        'seq', 
        'uniprot', 
        'taxon',
    ]

    def __init__(
        self, 
        file_path: str | Path,
        items: list | None = None,
        datasets: list[str | dict] | str | None = None,
        start: int | None = 0,
        end: int | None = None
    ):

        # Attributes
        self.file_path = Path(file_path)
        self.items = items
        self.start = start
        self.end = end
        if isinstance(datasets, list):
            self.datasets = self._pathify(datasets)
        else:
            self.datasets = datasets

        # Validation
        if self.end is not None and self.end <= self.start:
            raise ValueError("End index must be greater than start index.")

###############################################################################
#########                      CORE FUNCTIONS                         #########
###############################################################################

    def __len__(self) -> int:
        '''
        If the collection is loaded in memory, return the number of items in 
        the list. Otherwise, read the number of items from the HDF5 file
        attribute "num_items".

        Returns
        -------
        int
            Number of proteins in the collection.
        '''
        if self.items is not None:
            return len(self.items)

        with h5py.File(self.file_path, 'r') as file:
            return file.attrs["num_items"]

    def __iter__(self) -> Iterable[Protein]:
        '''
        If items are already in memory either because they were provided at 
        initialization or because they were previously loaded from HDF5, yield
        them directly. Otherwise, load them from the HDF5 file and yield them
        one by one. For this, the HDF5 file is expected to have a specific
        structure of groups and datasets corresponding to Protein attributes, 
        which are differentially loaded. 

        Yields
        ------
        Protein
            Protein objects in the collection.
        '''
        if self.items is not None:
            yield from self.items[self.start:self.end]
        
        with h5py.File(self.file_path, 'r') as file:
            num_items = len(self) if self.end is None else self.end
            desc = 'Loading Protein collection from HDF5 file'
            for idx in tqdm(range(self.start, num_items), desc=desc):
                yield self._load(file, idx)

###############################################################################
#########                         LOADING                             #########
###############################################################################

    def _load(
        self, 
        file: h5py.File, 
        idx: int
    ) -> Protein:
        '''
        Loads the contents of a single protein from the HDF5 file and returns
        the Protein object. 

        Parameters
        ----------
        file : h5py.File
            Open HDF5 file.
        idx : int
            Index of the protein in the HDF5 file.

        Returns
        -------
        Protein
            Protein object loaded from the HDF5 file.
        '''

        # Attributes
        kwargs = {}

        # Load standard datasets
        for dataset, dtype in self.DATASETS_DTYPES.items():
            # Skip if not requested
            if not self._should_load(dataset): continue
            if dataset not in file: continue
            # Access value
            if dtype != 'group':
                value = file[dataset][idx]
            # Decode
            match dtype:
                case "str":
                    kwargs[dataset] = value.decode('utf-8')
                case "int":
                    kwargs[dataset] = int(value)
                case "str_list":
                    kwargs[dataset] = [v.decode('utf-8') for v in value]
                case "json":
                    kwargs[dataset] = json.loads(value.decode('utf-8'))
                case "group":
                    if dataset == "esm2_embeddings":
                        kwargs[dataset] = self._load_esm2_embeddings(file, idx)
            
        return Protein(**kwargs)

    def _load_esm2_embeddings(
        self, 
        file: h5py.File, 
        idx: int
    ) -> dict:
        '''
        Loads ESM2 embeddings from the HDF5 file for a single protein.

        Parameters
        ----------
        file : h5py.File
            Open HDF5 file.
        idx : int
            Index of the protein in the HDF5 file.
        
        Returns
        -------
        dict
            Dictionary of ESM2 embeddings for the protein, with model names as 
            keys and embedding arrays as values.
        '''
        # Access group
        group = file["esm2_embeddings"]
        # Populate model-to-embedding dict
        embeddings = {}
        for model in group:
            dataset = f"esm2_embeddings/{model}"
            if not self._should_load(dataset): continue
            flatted = group[model][idx]
            dim = self.ESM_MODELS_DIMS.get(model)
            embeddings[model] = flatted.reshape(-1, dim)

        return embeddings

###############################################################################
#########                          SAVING                             #########
###############################################################################

    def _save(
        self, 
        h5: h5py.File | h5py.Group, 
        name: str,
        array: np.ndarray,
        dtype: str,
        ) -> None:
        '''
        Save an attribute to a HDF5 file or group.

        Parameters
        ----------
        h5 : h5py.File | h5py.Group
            Open HDF5 file or group.
        name : str
            Name of the dataset to be saved.
        array : np.ndarray
            Array to be saved.
        dtype : str
            Data type of the attribute to be saved.
        ''' 
        h5.create_dataset(
            name,
            data=array,
            dtype=dtype,
            compression="gzip", 
            compression_opts=9, 
            chunks=True
            )

    def to_hdf5(self) -> None:
        '''
        Save the ProteinCollection to an HDF5 file. Based on the attributes
        present in the Protein objects, and their types, its contents are saved
        accordingly. Therefore, it requires string matching for each attribute.
        Note: existing data is frozen and not overwritten.
        '''
        # Empty collection?
        if not self.items:
            raise ValueError("Cannot save an empty collection.")

        # Open HDF5 file in append mode
        with h5py.File(self.file_path, "a") as file:

            # Number of items as attribute
            file.attrs['num_items'] = len(self.items)

            # Iterate over attributes
            for attr, dtype in self.DATASETS_DTYPES.items():

                # Skip existing datasets
                if attr in file:
                    continue
                # Skip attributes with all default values
                all_default = all(
                    getattr(item, attr) in (None, {}, []) 
                    for item in self.items
                )
                if all_default: continue
                # Save non-default attributes
                match dtype:
                    
                    # String attributes
                    case "str":
                        dt = h5py.string_dtype(encoding='utf-8')
                        values = [getattr(item, attr) for item in self.items]
                        array = np.array(values, dtype=dt)
                        self._save(file, attr, array, dt)
                    # Integer attributes
                    case "int":
                        dt = np.int32
                        values = [getattr(item, attr) for item in self.items]
                        array = np.array(values, dtype=dt)
                        self._save(file, attr, array, dt)
                    # List of strings attributes
                    case "str_list":
                        lst = [getattr(item, attr) for item in self.items]
                        dt = h5py.vlen_dtype(h5py.string_dtype(encoding='utf-8'))
                        lst = [np.array(sublist, dtype=dt) for sublist in lst]
                        array = np.array(lst, dtype=dt)
                        self._save(file, attr, array, dt)
                    # JSON attributes
                    case "json":
                        dt = h5py.string_dtype(encoding='utf-8')
                        lst = [
                            json.dumps(getattr(item, attr)) 
                            for item in self.items
                        ]
                        array = np.array(lst, dtype=dt)
                        self._save(file, attr, array, dt)
                    # Special attributes (e.g. embeddings)
                    case "group":
                        match attr:
                            case "esm2_embeddings":
                                self._save_esm2_embeddings(file)

    def _save_esm2_embeddings(
        self, 
        file: h5py.File,
        ) -> None:
        '''
        Save ESM2 embeddings to a HDF5 file. Since there are multiple models,
        each with different dimensions, they are saved in separate groups under
        the main group "esm2_embeddings". Each model group contains a dataset
        with the corresponding embeddings.

        Parameters
        ----------
        file : h5py.File
            Open HDF5 file.
        '''
        # Create main group
        group = file.create_group('esm2_embeddings')
        
        # Iterate over models and save embeddings
        for model in self.items[0].esm2_embeddings.keys():
            embeddings = [item.esm2_embeddings[model] for item in self.items]
            embeddings = [
                (
                    embedding.flatten()
                    if embedding is not None
                    else np.array([], dtype=np.float32)
                )
                for embedding in embeddings
            ] # vlen supports 1D only
            dt = h5py.vlen_dtype(np.dtype('float32'))
            array = np.array(embeddings, dtype=dt)
            group = self._save(group, model, array, dt)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class ProteinPairCollection(Collection):

    DATASETS_DTYPES = {
        "p1": "protein",
        "p2": "protein",
        "bind": "float",
        "mi_score": "float",
    }

    LIGHTWEIGHT = [
        'p1',
        'p2',
        'bind'
    ]

    def __init__(
        self, 
        file_path: str | Path,
        proteins: list[Protein] | ProteinCollection | str | Path | None = None,
        items: list | None = None,
        datasets: list[str | dict] | str | None = None,
        start: int | None = 0,
        end: int | None = None
        ):

        # Attributes
        self.file_path = Path(file_path)
        self.proteins = self._load_proteins(proteins)
        self.items = items
        self.start = start
        self.end = end
        if isinstance(datasets, list):
            self.datasets = self._pathify(datasets)
        else:
            self.datasets = datasets

        # Validation
        if self.end is not None and self.end <= self.start:
            raise ValueError("End index must be greater than start index.")     


###############################################################################
#########                      CORE FUNCTIONS                         #########
###############################################################################

    def __len__(self) -> int:
        '''
        If the collection is loaded in memory, return the number of items in 
        the list. Otherwise, read the number of items from the HDF5 file
        attribute "num_items".

        Returns
        -------
        int
            Number of proteins in the collection.
        '''
        if self.items is not None:
            return len(self.items)

        with h5py.File(self.file_path, 'r') as file:
            return file.attrs["num_items"]

    def __iter__(self) -> Iterable[ProteinPair]:
        '''
        If items are already in memory either because they were provided at 
        initialization or because they were previously loaded from HDF5, yield
        them directly. Otherwise, load them from the HDF5 file and yield them
        one by one. For this, the HDF5 file is expected to have a specific
        structure of groups and datasets corresponding to PPI attributes, 
        which are differentially loaded. 

        Yields
        ------
        ProteinPair
            ProteinPair objects in the collection.
        '''
        if self.items is not None:
            yield from self.items[self.start:self.end]
        
        with h5py.File(self.file_path, 'r') as file:
            num_items = len(self) if self.end is None else self.end
            desc = 'Loading ProteinPair collection from HDF5 file'
            for idx in tqdm(range(self.start, num_items), desc=desc):
                yield self._load(file, idx)

###############################################################################
#########                         LOADING                             #########
###############################################################################

    def _load_proteins(
        self, 
        proteins: list[Protein]  | ProteinCollection | str | Path | None = None
        ) -> list[Protein]:
        '''
        Loads Protein objects from a list, a file path, or a ProteinCollection.

        Parameters
        ----------
        proteins : list[Protein] | ProteinCollection | str | Path | None
            List of Protein objects, path to a file containing Protein objects, 
            or None if proteins are not to be loaded.
        
        Returns
        -------
        list[Protein]
            List of loaded Protein objects.
        '''
        if proteins is None:
            return None
        if isinstance(proteins, list):
            return proteins
        if isinstance(proteins, (str, Path)):
            protein_collection = ProteinCollection(
                file_path=proteins,
                datasets='lightweight'
            )
            proteins = [protein for protein in protein_collection]
            return proteins
        if isinstance(proteins, ProteinCollection):
            proteins = [protein for protein in proteins]
            return proteins
        else:
            raise ValueError("Invalid input for proteins. Must be a list, a file path, or None.")

    def _load(
        self, 
        file: h5py.File, 
        idx: int
    ) -> ProteinPair:
        '''
        Loads the contents of a single protein pair from the HDF5 file and 
        returns the ProteinPair object. 

        Parameters
        ----------
        file : h5py.File
            Open HDF5 file.
        idx : int
            Index of the PPI in the HDF5 file.

        Returns
        -------
        ProteinPair
            ProteinPair object loaded from the HDF5 file.
        '''
        # Attributes
        kwargs = {}

        # Load standard datasets
        for dataset, dtype in self.DATASETS_DTYPES.items():
            # Skip if not requested
            if not self._should_load(dataset): continue
            if dataset not in file: continue
            # Access value
            if dtype != 'group':
                value = file[dataset][idx]
            # Decode
            match dtype:
                case "protein":
                    kwargs[dataset] = self.proteins[int(value)]
                case "str":
                    kwargs[dataset] = value.decode('utf-8')
                case "str_list":
                    kwargs[dataset] = [v.decode('utf-8') for v in value]
                case "int":
                    kwargs[dataset] = int(value)
                case "float":
                    kwargs[dataset] = float(value)
                case "json":
                    kwargs[dataset] = json.loads(value.decode('utf-8'))

        return ProteinPair(**kwargs)


###############################################################################
#########                          SAVING                             #########
###############################################################################

    def _save(
        self, 
        h5: h5py.File | h5py.Group, 
        name: str,
        array: np.ndarray,
        dtype: str,
    ) -> None:
        '''
        Save an attribute to a HDF5 file or group.

        Parameters
        ----------
        h5 : h5py.File | h5py.Group
            Open HDF5 file or group.
        name : str
            Name of the dataset to be saved.
        array : np.ndarray
            Array to be saved.
        dtype : str
            Data type of the attribute to be saved.
        ''' 
        h5.create_dataset(
            name,
            data=array,
            dtype=dtype,
            compression="gzip", 
            compression_opts=9, 
            chunks=True
            )

    def to_hdf5(
        self,
        not_skip: str | list[str] | None = None
    ) -> None:
        '''
        Save the PPICollection to an HDF5 file. Based on the attributes
        present in the Protein objects, and their types, its contents are saved
        accordingly. Therefore, it requires string matching for each attribute.
        Note: existing data is frozen and not overwritten.

        Parameters
        ----------
        not_skip : str | list[str] | None
            Dataset or list of datasets that should be saved even if they
            already exist in the HDF5 file. This is useful for updating specific
            datasets without overwriting the entire file.
        '''
        # Empty collection?
        if not self.items:
            raise ValueError("Cannot save an empty collection.")

        # Open HDF5 file in append mode
        with h5py.File(self.file_path, "a") as file:

            # Number of items as attribute
            file.attrs['num_items'] = len(self.items)

            # Iterate over attributes
            for attr, dtype in self.DATASETS_DTYPES.items():
                # Skip existing datasets
                if attr in file and (not_skip is None or attr not in not_skip):
                    continue
                # Skip attributes with all default values
                all_default = all(
                    self._is_default(getattr(item, attr))
                    for item in self.items
                )
                if all_default: continue
                # Save non-default attributes
                match dtype:
                    
                    # Handle protein attributes
                    case "protein":
                        prot2idx = {
                            prot: idx
                            for idx, prot in enumerate(self.proteins)
                        }
                        prots = [getattr(item, attr) for item in self.items]
                        indices = [prot2idx[prot] for prot in prots]
                        dt = np.int32
                        array = np.array(indices, dtype=dt)
                        self._save(file, attr, array, dt)
                    # String attributes
                    case "str":
                        dt = h5py.string_dtype(encoding='utf-8')
                        values = [getattr(item, attr) for item in self.items]
                        array = np.array(values, dtype=dt)
                        self._save(file, attr, array, dt)
                    # List of strings attributes
                    case "str_list":
                        lst = [getattr(item, attr) for item in self.items]
                        dt = h5py.vlen_dtype(h5py.string_dtype(encoding='utf-8'))
                        lst = [np.array(sublist, dtype=dt) for sublist in lst]
                        array = np.array(lst, dtype=dt)
                        self._save(file, attr, array, dt)
                    # Integer attributes
                    case "int":
                        dt = np.int32
                        values = [getattr(item, attr) for item in self.items]
                        array = np.array(values, dtype=dt)
                        self._save(file, attr, array, dt)
                    # Float attributes
                    case "float":
                        dt = np.float32
                        values = [getattr(item, attr) for item in self.items]
                        array = np.array(values, dtype=dt)
                        self._save(file, attr, array, dt)
                    # JSON attributes
                    case "json":
                        dt = h5py.string_dtype(encoding='utf-8')
                        lst = [
                            json.dumps(getattr(item, attr)) 
                            for item in self.items
                        ]
                        array = np.array(lst, dtype=dt)
                        self._save(file, attr, array, dt)