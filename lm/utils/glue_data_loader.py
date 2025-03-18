import os

import datasets
import numpy as np
from torch.utils.data import Subset, Dataset
import transformers

from utils.load_config import cache_dir


glue_data_keys_map = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2")
}

glue_data_metrics_map = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "averaged_scores",   
    "stsb": "averaged_scores",   
    "qqp": "averaged_scores",    
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy"
}

glue_data_num_labels_map = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2
}

glue_data_id_map = {
    "cola": 0,
    "sst2": 1,
    "mrpc": 2,
    "stsb": 3,
    "qqp": 4,
    "mnli": 5,
    "qnli": 6,
    "rte": 7
}

rev_glue_data_id_map = {value: key for key, value in glue_data_id_map.items()}


class GLUEDataLoader:
    def __init__(self, tokenizer: transformers.AutoTokenizer):
        self.tokenizer = tokenizer

    def load_dataset(self, dataset_name: str, train_split_ratio_for_val: float = 0.1, max_seq_length: int = 128):
        sub_task_dir = os.path.join(cache_dir, dataset_name)
        dataset = datasets.load_from_disk(sub_task_dir)

        sentence1_key, sentence2_key = glue_data_keys_map[dataset_name]


        dataset = dataset.map(lambda examples: self.tokenizer(text=examples[sentence1_key],
                                                              text_pair=examples[sentence2_key] if sentence2_key else None,
                                                              max_length=max_seq_length, truncation=True),
                              batched=True)

        dataset = dataset.map(lambda x: {"dataset_ids": glue_data_id_map[dataset_name]})

        permuted_indices = np.random.RandomState(seed=0).permutation(len(dataset["train"])).tolist()
        num_train_data = int((1 - train_split_ratio_for_val) * len(dataset["train"]))
        train_dataset = Subset(dataset=dataset["train"], indices=permuted_indices[:num_train_data])

        val_dataset = Subset(dataset=dataset["train"], indices=permuted_indices[num_train_data:])
        test_dataset = dataset["validation_matched"] if dataset_name == "mnli" else dataset["validation"]
        num_labels = glue_data_num_labels_map[dataset_name]

        return train_dataset, val_dataset, test_dataset, num_labels

    def load_multitask_datasets(self, dataset_names: list, train_split_ratio_for_val: float = 0.1, max_seq_length: int = 128):
        assert isinstance(dataset_names, list) and len(dataset_names) > 1, f"wrong setting on datasets {dataset_names}!"
        multiple_datasets = [self.load_dataset(dataset_name=dataset_name, train_split_ratio_for_val=train_split_ratio_for_val,
                                               max_seq_length=max_seq_length)
                             for dataset_name in dataset_names]
        train_datasets, val_datasets, test_datasets, datasets_num_labels = [dataset[0] for dataset in multiple_datasets], \
            [dataset[1] for dataset in multiple_datasets], [dataset[2] for dataset in multiple_datasets], [dataset[3] for dataset in multiple_datasets]

        multi_train_datasets = MultiDatasets(datasets=train_datasets)
        multi_val_datasets = MultiDatasets(datasets=val_datasets)
        multi_test_datasets = MultiDatasets(datasets=test_datasets)

        return multi_train_datasets, multi_val_datasets, multi_test_datasets, datasets_num_labels


class MultiDatasets(Dataset):
    def __init__(self, datasets: list):
        super(MultiDatasets, self).__init__()
        self.datasets = datasets

    def __getitem__(self, index: int):
        dataset_index = 0
        assert 0 <= index < len(self), f"index {index} out of the length of data {len(self)}!"
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index][index]

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])