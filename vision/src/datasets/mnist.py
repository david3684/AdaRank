import os
import torch
import torchvision.datasets as datasets


class MNIST:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 persistent_workers=False,
                 subset_ratio=1,
                 ):

        self.train_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )

        from torch.utils.data import Subset
        import random
        dataset_size = len(self.test_dataset)
        subset_size = int(0.1 * dataset_size)

        
        indices = list(range(dataset_size))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]

        if subset_ratio != 1:
            test_subset = Subset(self.test_dataset, subset_indices)
            self.test_loader = torch.utils.data.DataLoader(
                test_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True
            )
        else:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, num_workers=num_workers
            )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            persistent_workers=persistent_workers,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
