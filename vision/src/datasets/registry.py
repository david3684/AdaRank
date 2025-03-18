import os
import sys
import inspect
import random
import torch
import copy


repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from torch.utils.data.dataset import random_split

from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.gtsrb import GTSRB
from src.datasets.imagenet import ImageNet
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.stl10 import STL10
from src.datasets.svhn import SVHN
from src.datasets.sun397 import SUN397

from src.datasets.cifar100 import CIFAR100
from src.datasets.stl10 import STL10
from src.datasets.flowers102 import Flowers102
from src.datasets.oxfordpets import OxfordIIITPet
from src.datasets.pcam import PCAM
from src.datasets.fer2013 import FER2013

from src.datasets.emnist import EMNIST
from src.datasets.cifar10 import CIFAR10
from src.datasets.food101 import Food101
from src.datasets.fashionmnist import FashionMNIST
from src.datasets.sst2 import RenderedSST2
from src.datasets.kmnist import KMNIST


registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}

DATASET_MAP = {
    "Cars": Cars,
    "DTD": DTD,
    "EuroSAT": EuroSAT,
    "GTSRB": GTSRB,
    "MNIST": MNIST,
    "RESISC45": RESISC45,
    "SUN397": SUN397,
    "SVHN": SVHN,
    "CIFAR100": CIFAR100,
    "STL10": STL10,
    "Flowers102": Flowers102,
    "OxfordIIITPet": OxfordIIITPet,
    "PCAM": PCAM,
    "FER2013": FER2013,
    "EMNIST": EMNIST,
    "CIFAR10": CIFAR10,
    "Food101": Food101,
    "FashionMNIST": FashionMNIST,
    "RenderedSST2": RenderedSST2,
    "KMNIST": KMNIST,
}

tasks_8 = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
] 

addition_6_datasets_split1 = [
    "CIFAR100",  # done
    "STL10",
    "Flowers102",
    "OxfordIIITPet",
    "PCAM",
    "FER2013",
]

addition_6_datasets_split2 = [
    "EMNIST",
    "CIFAR10",
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "KMNIST",
]

tasks_14 = tasks_8 + addition_6_datasets_split1
tasks_20 = tasks_14 + addition_6_datasets_split2


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16, val_fraction=0.1, max_val_samples=5000, subset_ratio=1, persistent_workers=False):
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(
                base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]

    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, subset_ratio = subset_ratio
    )
    return dataset


if __name__ == "__main__":
    print(registry)
