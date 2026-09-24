"""
===============================================================================
Title:      Split
Outline:    Split class to split a dataset into train, validation, and test
            sets in a simple or k-fold manner.
Author:     Alejandro Sánchez Cano
Date:       23/09/2026
===============================================================================
"""

# Built-in modules
import random

# Third-party modules
import math
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset, random_split

# Custom modules
from misc.logger import logger

class Split:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def simple(
        self, 
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1
    ) -> tuple[Dataset, Dataset, Dataset]:
        '''
        Simple split of the dataset into train, validation, and test sets.

        Parameters
        ----------
        train_size : float, optional
            Size of the train set, by default 0.8.
        val_size : float, optional
            Size of the validation set, by default 0.1.
        test_size : float, optional
            Size of the test set, by default 0.1.

        Returns
        -------
        tuple[Dataset, Dataset, Dataset]
            Train, validation, and test datasets.
        '''
        # Validate sizes
        assert abs(
            train_size + val_size + test_size - 1
            ) < 0.001 , "Sizes must sum up to 1"

        # Datsaset sizes
        train_size = math.ceil(train_size * len(self.dataset))
        val_size = math.ceil(val_size * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

        # Logging
        logger.info(f'Train size: {len(train_dataset)}')
        logger.info(f'Validation size: {len(val_dataset)}')
        logger.info(f'Test size: {len(test_dataset)}')

        return train_dataset, val_dataset, test_dataset

    def kfold(
        self,
        k: int,
        cv_size: float = 0.9,
        test_size: float = 0.1
        ) -> tuple[list[Dataset], list[Dataset], Dataset]:
        '''
        K-fold split of the dataset into train, validation, and test sets.

        Parameters
        ----------
        k : int
            Number of folds for the k-fold split.
        cv_size : float, optional
            Size of the train + validation set, by default 0.9.
        test_size : float, optional
            Size of the test set, by default 0.1.

        Returns
        -------
        tuple[list[Dataset], list[Dataset], Dataset]
            Train, validation, and test datasets.
        '''
        # Validate sizes
        assert abs(
            cv_size + test_size - 1
            ) < 0.001 , "Sizes must sum up to 1"

        # Dataset sizes
        train_val_size = math.ceil(cv_size * len(self.dataset))
        test_size = len(self.dataset) - train_val_size

        # Split dataset
        train_val_dataset, test_dataset = random_split(
            self.dataset, [train_val_size, test_size]
        )

        # KFold train and validation sets
        train_val_indices = train_val_dataset.indices
        train_datasets, val_datasets = [], []
        kf = KFold(n_splits=k, shuffle=True)
        for train_idx, val_idx in kf.split(train_val_indices):
            train_subset = Subset(self.dataset, [train_val_indices[i] for i in train_idx])
            val_subset = Subset(self.dataset, [train_val_indices[i] for i in val_idx])
            train_datasets.append(train_subset)
            val_datasets.append(val_subset)

        # Logging
        logger.info(f'Train + validation size: {len(train_subset) + len(val_subset)} divided in {k} folds')
        logger.info(f'Each fold has train size: {len(train_subset)} and validation size: {len(val_subset)}')
        logger.info(f'Test size: {len(test_dataset)}')

        return train_datasets, val_datasets, test_dataset

    def manual(
        self, 
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int]
    ) -> tuple[Dataset, Dataset, Dataset]:
        '''
        Manual split of the dataset into train, validation, and test sets based 
        on cluster membership.

        Parameters
        ----------
        train_indices : list[int]
            List of indices for the train set.
        val_indices : list[int]
            List of indices for the validation set.
        test_indices : list[int]
            List of indices for the test set.

        Returns
        -------
        tuple[Dataset, Dataset, Dataset]
            Train, validation, and test datasets.
        '''
        # Create subsets
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)

        # Logging
        logger.info(f'Train size: {len(train_dataset)}')
        logger.info(f'Validation size: {len(val_dataset)}')
        logger.info(f'Test size: {len(test_dataset)}')

        return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    # Example usage
    from torch.utils.data import Dataset

    class ExampleDataset(Dataset):
        def __init__(self, size):
            self.data = list(range(size))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    print('Testing Split class...')

    dataset = ExampleDataset(100)

    split = Split(
        dataset=dataset
    )
    train, val, test = split.simple(
        train_size=0.7,
        val_size=0.2,
        test_size=0.1
    )

    train, val, test = split.kfold(
        k=2,
        cv_size=0.8,
        test_size=0.2
    )
    
    train_indices = list(range(30))
    val_indices = list(range(70, 90))
    test_indices = list(range(90, 90))
    datasets = split.manual(
        train_indices=train_indices, 
        val_indices=val_indices, 
        test_indices=test_indices
    )
