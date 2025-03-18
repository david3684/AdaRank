from dataclasses import dataclass, field
from typing import Dict, Optional

import os
import pandas as pd
import namesgenerator
from typing import Optional
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe

from src.datasets.registry import tasks_8, tasks_14, tasks_20

LOGGING_TARGET_HYPS =[
    "merge_type",
    "merge_method",
    "num_tasks",
    "initial_rank_ratio",
    "prior",
    "initial_merge_coeff",
    "initial_merge_rank_ratio",
]

@dataclass
class ExperimentResult:
    method: str
    merge_type: str
    num_tasks: int
    scores: Dict[str, float] = field(default_factory=dict)
    index: Optional[str] = None
    exp_config: Optional[Dict] = None

    def __post_init__(self):
        if self.index is None:
            self.index = namesgenerator.get_random_name()
        
        if self.num_tasks == 8:
            _tasks = tasks_8
        elif self.num_tasks == 14:
            _tasks = tasks_14
        else:
            _tasks = tasks_20
        
        for task in _tasks:
            self.scores[task] = 0.0

    def add_score(self, task: str, score: float):
        assert task in self.scores.keys(), f"{task} is not in the predefined task list."
        self.scores[task] = score

    def get_score(self, task: str):
        return self.scores.get(task, None)

    def _get_avg_score(self):
        avg_score = sum(self.scores.values()) / len(self.scores)
        return avg_score

    def to_dict(self):
        base = {}
        if self.exp_config is not None:
            for key in LOGGING_TARGET_HYPS:
                _value = self.exp_config.get(key, None)
                if _value is not None:
                    base[key] = str(_value)
        else:
            base = {
                "index": self.index,
                "num_tasks": self.num_tasks,
                "method": self.method,
                "merge_type": self.merge_type,
            }
        for task in self.scores.keys():
            base[task] = self.scores[task]
        base["avg"] = self._get_avg_score()
        return base


def csv_logger(exp_result: ExperimentResult, exp_name: str):
    df = pd.DataFrame([exp_result.to_dict()])
    
    exp_dir = os.path.join("/workspace/")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, f"{exp_name}.csv"), index=False)
    
    


    