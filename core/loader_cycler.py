"""
===============================================================================
Title:      Loder Cycler
Outline:    LoaderCycler class to cycle through multiple DataLoaders for 
            multi-task learning scenarios. It ensures that each DataLoader is 
            iterated over as long as the longest DataLoader is not exhausted.
Author:     Alejandro Sánchez Cano
Date:       09/10/2026
===============================================================================
"""

class LoaderCycler:

    def __init__(self, loaders: list):
        self.loaders = loaders

        # Validate that all DataLoaders have drop_last=True
        for loader in self.loaders:
            if not loader.drop_last:
                raise ValueError(
                    "All DataLoaders must have drop_last=True to ensure "
                    "consistent batch sizes across tasks."
                )

    def __len__(self):
        '''Return the length of the longest DataLoader.'''
        return max(len(loader) for loader in self.loaders)

    def __iter__(self):
        '''Return an iterator over the cycled batches.'''
        # Create an iterator for each DataLoader
        iterators = [iter(loader) for loader in self.loaders]

        # Cycle through the DataLoaders until the longest one is exhausted
        for _ in range(len(self)):
            batches = []
            for idx, loader in enumerate(self.loaders):
                # Get next batch
                try: 
                    batch = next(iterators[idx])
                # Restart iterator if exhausted
                except StopIteration:
                    iterators[idx] = iter(loader)
                    batch = next(iterators[idx])

                batches.append(batch)
            
            yield batches

if __name__ == "__main__":

    # Test imports
    import torch

    # Example dataset
    class ExampleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create multiple datasets
    task1 = ExampleDataset(data=[1, 2, 3, 4, 5, 6, 7, 8])
    task2 = ExampleDataset(data=[10, 20])
    task3 = ExampleDataset(data=[100, 200, 300])

    # Create DataLoaders for each dataset
    loader1 = torch.utils.data.DataLoader(
        task1, batch_size=2, shuffle=True, drop_last=True
    )
    loader2 = torch.utils.data.DataLoader(
        task2, batch_size=2, shuffle=True, drop_last=True
    )
    loader3 = torch.utils.data.DataLoader(
        task3, batch_size=2, shuffle=True, drop_last=True
    )

    # Create a cycler
    cycler = LoaderCycler([loader1, loader2, loader3])

    # Iterate through the cycler
    for idx, batches in enumerate(cycler):
        print(f"Batches {idx}: {batches}")