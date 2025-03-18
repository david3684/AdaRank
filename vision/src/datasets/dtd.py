import os
import torch
import torchvision.datasets as datasets
from torchvision.datasets import DTD as TorchvisionDTD
from torch.utils.data.sampler import SubsetRandomSampler

class DTD:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=16,
        persistent_workers=False,
        subset_ratio=1,
    ):  

        # Data loading code
        traindir = os.path.join(location, "dtd", "train")
        valdir = os.path.join(location, "dtd", "val")

        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        from torch.utils.data import Subset
        import random
        dataset_size = len(self.test_dataset)
        subset_size = int(subset_ratio * dataset_size)

        
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
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [
            idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))
        ]
