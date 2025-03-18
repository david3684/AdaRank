import torch
import torch.nn as nn
from copy import deepcopy
from transformers import AutoModelForSequenceClassification
from transformers import (
    GPT2ForSequenceClassification,
)
from utils.load_config import cache_dir, weight_dir
import os
from tqdm import tqdm
from collections import defaultdict
from model_merging_methods.task_vector import TaskVector
from utils.utils import get_param_names_to_merge

ARCH_NAME_LIST = [
    "roberta-base",
    "gpt2"
]

CPU_DEVICE = "cpu"


ROBERTA_EXCLUDE_PARAM_NAMES_REGEX_AVG = [".*classifier.*"]
ROBERTA_EXCLUDE_PARAM_NAMES_REGEX_tv = [".*classifier.*", ".*embeddings*."]


GPT_EXCLUDE_PARAM_NAMES_REGEX_AVG = [".*score.*"]
GPT_EXCLUDE_PARAM_NAMES_REGEX_tv = [".*score.*", ".*wte*.", ".*wpe*."]

WEIGHT_DIR: str = weight_dir + "gpt2"

EXCLUDE_MAP = {
    "roberta-base": {
        "avg": ROBERTA_EXCLUDE_PARAM_NAMES_REGEX_AVG,
        "tv": ROBERTA_EXCLUDE_PARAM_NAMES_REGEX_tv
    },
    "gpt2": {
        "avg": GPT_EXCLUDE_PARAM_NAMES_REGEX_AVG,
        "tv": GPT_EXCLUDE_PARAM_NAMES_REGEX_tv
    }
}

def is_mat_params(arch_name: str, param_name: str) -> bool:
    assert arch_name in ARCH_NAME_LIST, f"arch_name should be one of {ARCH_NAME_LIST}, currently {arch_name}"
    if arch_name == "roberta-base":
        if not (
            "bias" in param_name or "LayerNorm" in param_name or "token_type_embeddings" in param_name
        ):
            return True
        else:
            return False
    elif arch_name == "gpt2":
        if not ("ln" in param_name or "bias" in param_name):
            return True
        else:
            return False
    else:
        raise NotImplementedError(f"wrong arch_name: {arch_name}")

def get_exclude_regex(arch_name: str, deompose: str):
    if deompose == "full":
        return EXCLUDE_MAP[arch_name]["avg"]
    elif deompose == "encoder":
        return EXCLUDE_MAP[arch_name]["tv"]
    else:
        return None

class AdaRankModule(nn.Module):
    def __init__(self, config, models_to_merge):
        super(AdaRankModule, self).__init__()
        self.config = config
        self.models_to_merge = models_to_merge

        if self.config.merge_backbone == "roberta-base":
            try:
                pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=os.path.join(cache_dir, self.config.merge_backbone)).to(self.config.device)
            except:
                pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=self.config.merge_backbone, cache_dir=cache_dir).to(self.config.device)
        else:
            pretrained_model = GPT2ForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=os.path.join(WEIGHT_DIR, "gpt2")).to(self.config.device)
        
        self.base_model = deepcopy(pretrained_model)
        self.origin = pretrained_model

        pretrained_param_dict = {param_name: param_value for param_name,
                                     param_value in self.base_model.named_parameters()}
        
        _exclude_regex = get_exclude_regex(
                self.config.merge_backbone, self.config.decompose)
        
        self.param_names_to_merge = get_param_names_to_merge(input_param_names=list(
                pretrained_param_dict.keys()), exclude_param_names_regex=_exclude_regex)
        
        if config.merge_method == 'cart':
            avg_model = pretrained_model
            avg_origin = self.average_merging(
                models_to_merge=models_to_merge,
                exclude_param_names_regex=EXCLUDE_MAP[config.merge_backbone]["avg"]
            )
            self.copy_params_to_model(avg_origin, avg_model)
            self.base_model = deepcopy(avg_model.to(config.device))
            self.origin = avg_model
            print("Origin has been changed to average")
        
        print("Task Vector Initialization")
        self.models_to_merge_task_vectors = []
        for each_model in tqdm(models_to_merge):

            _exclude_regex = get_exclude_regex(
                config.merge_backbone, config.decompose)
            
            each_tv = TaskVector(
                    pretrained_model=self.base_model,
                    finetuned_model=each_model,
                    exclude_param_names_regex=_exclude_regex,
                    do_svd_truncation=False,
            )

            self.models_to_merge_task_vectors.append(each_tv)
       
        self.svd_list = self._svd_vanilla(config, self.models_to_merge_task_vectors)

        merge_mask = self.mask_init(config)
        rlambda = torch.ones(len(self.models_to_merge), len(self.param_names_to_merge), device=config.device) * config.prior
        self.merge_weight = nn.Parameter(rlambda)


        self.merge_mask = nn.ParameterList(merge_mask)


        self.mask_temp = config.mask_temp

        self._overall_requires_grad()

    def get_model(self):
        self.reset_merged_state()
        self._merge_weights()
        if self.config.merge_backbone == "roberta-base":
            try:
                pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=os.path.join(cache_dir, self.config.merge_backbone)).to(self.config.device)
            except:
                pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=self.config.merge_backbone, cache_dir=cache_dir).to(self.config.device)
        else:
            pretrained_model = GPT2ForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=os.path.join(WEIGHT_DIR, "gpt2")).to(self.config.device)
        self.copy_params_to_model(self._merged_state_dict, pretrained_model)
        return pretrained_model
    
    def _merge_weights(self):
        task_vectors = self.models_to_merge_task_vectors
        config = self.config
        base_state = {param_name: param_value.clone() for param_name,
                      param_value in self.origin.named_parameters()}
        layer_wise_weight = self.merge_weight
        
        for idx, (weight, svd) in enumerate(zip(layer_wise_weight, self.svd_list)):
            for m, w, param_name in zip(self.merge_mask, weight, self.param_names_to_merge):
                if is_mat_params(config.merge_backbone, param_name):
                    tv = self.straight_through_mask(svd[param_name], (m[idx] / self.mask_temp).sigmoid())
                    base_state[param_name] = base_state[param_name] + w * tv
                else:
                    base_state[param_name] = base_state[param_name] + w * svd[param_name]

        self._merged_state_dict = base_state
    
    def _svd_vanilla(self, config, task_vectors):
        svd_list = []
        for task_vector in task_vectors:
            svd_dict = {}
            for param_name in self.param_names_to_merge:
                if is_mat_params(config.merge_backbone, param_name):
                    svd_dict[param_name] = torch.linalg.svd(task_vector.task_vector_param_dict[param_name], full_matrices=False)
                else:
                    svd_dict[param_name] = task_vector.task_vector_param_dict[param_name]

            svd_list.append(svd_dict)

        return svd_list
    
    def _overall_requires_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if "merge_weight" in name or "merge_mask" in name:
                param.requires_grad = True

    def mask_init(self, config):
        merge_mask = []
        merge_weight = []
        pretrained_param_dict = {param_name: param_value for param_name,
                                param_value in self.base_model.named_parameters()}
        for param_name in tqdm(self.param_names_to_merge):
            if is_mat_params(config.merge_backbone, param_name):
                dim = min(pretrained_param_dict[param_name].shape[0], pretrained_param_dict[param_name].shape[1])
                mask = 0.1 * torch.ones(len(self.models_to_merge), dim, device=config.device)

                preserved_dim = int(dim * self.config.initial_rank_ratio)
                mask[:, preserved_dim:] = -0.1

                merge_mask.append(torch.nn.Parameter(mask, requires_grad=True))
            else:
                merge_mask.append(torch.nn.Parameter(torch.ones(1, device=config.device)))
        return merge_mask
    
    def reset_merged_state(self):
        self._merged_state_dict = None
        
    def average_merging(self, models_to_merge: list, exclude_param_names_regex: list):
        models_to_merge_param_dict = defaultdict(list)
        for model_to_merge in models_to_merge:
            param_dict = {param_name: param_value for param_name,
                          param_value in model_to_merge.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(
                param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(
                    param_dict[param_name])

        with torch.no_grad():
            averaged_params = {param_name: torch.stack(model_to_merge_param, dim=0).mean(
                dim=0) for param_name, model_to_merge_param in models_to_merge_param_dict.items()}

        return averaged_params
    
    def copy_params_to_model(self, params: dict, model: nn.Module):
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def straight_through_mask(self, mat_svd, mask):
        U, s, V_T = mat_svd
        s_masked = mask * s + (((mask > 0.5).float() - mask) * s).detach()
        return U @ torch.diag(s_masked) @ V_T

    from torch.nn.utils import stateless

    def forward(self, **inputs):
        if self._merged_state_dict is None:
            self._merge_weights()
        
        dataset_ids = inputs.pop("dataset_ids", None)
        if dataset_ids is None:
            raise ValueError("Input dictionary must contain 'dataset_ids' key.")
        if isinstance(dataset_ids, torch.Tensor):
            dataset_id = int(dataset_ids[0].item())
        else:
            dataset_id = int(dataset_ids)
        idx = dataset_id
        
        if self.config.merge_backbone == "roberta-base":
            classifier_state = self.models_to_merge[idx].classifier.state_dict()
            for key, value in classifier_state.items():
                self._merged_state_dict[f"classifier.{key}"] = value

        else:
            classifier_state = self.models_to_merge[idx].score.state_dict()
            for key, value in classifier_state.items():
                self._merged_state_dict[f"score.{key}"] = value

        out = torch.nn.utils.stateless.functional_call(self.base_model, self._merged_state_dict, (), inputs)
        return out