from collections import defaultdict, OrderedDict
from typing import Optional
from tqdm import tqdm
import copy
import torch
import torch.nn as nn

from model_merging_methods.task_vector import TaskVector
from utils.utils import get_param_names_to_merge

from omegaconf import OmegaConf, DictConfig
from copy import deepcopy

ROBERTA_EXCLUDE_PARAM_NAMES_REGEX_AVG = [".*classifier.*"]
ROBERTA_EXCLUDE_PARAM_NAMES_REGEX_tv = [".*classifier.*", ".*embeddings*."]

GPT_EXCLUDE_PARAM_NAMES_REGEX_AVG = [".*score.*"]
GPT_EXCLUDE_PARAM_NAMES_REGEX_tv = [".*score.*", ".*wte*.", ".*wpe*."]

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


def get_exclude_regex(arch_name: str, deompose: str):
    if deompose == "full":
        return EXCLUDE_MAP[arch_name]["avg"]
    elif deompose == "encoder":
        return EXCLUDE_MAP[arch_name]["tv"]
    else:
        return None


class MergingMethod:
    def __init__(self, merging_method_name: str):
        self.merging_method_name = merging_method_name

    def copy_params_to_model(self, params: dict, model: nn.Module):
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def cart_merging(self, prev_origin, models_to_merge, exclude_param_names_regex, merge_config: DictConfig):
        assert merge_config.merge_rank_ratio is not None, f"current value of merge_rank_ratio is {merge_config.merge_rank_ratio}"
        assert merge_config.merge_coeff is not None, f"current value of merge_coeff is {merge_config.merge_coeff}"

        print("Construct Average model w/o head")
        avg_model: nn.Module = deepcopy(prev_origin)

        avg_origin = self.average_merging(
            models_to_merge=models_to_merge,
            exclude_param_names_regex=EXCLUDE_MAP[merge_config.merge_backbone]["avg"]
        )
        avg_model.load_state_dict(avg_origin, strict=False)
        
        print(
            f"Task Vector truncation rank ratio: {merge_config.merge_rank_ratio}")

        models_to_merge_task_vectors = []
        for each_model in tqdm(models_to_merge):

            _exclude_regex = get_exclude_regex(
                merge_config.merge_backbone, merge_config.decompose)
            each_tv = TaskVector(
                pretrained_model=avg_model,
                finetuned_model=each_model,
                exclude_param_names_regex=_exclude_regex,
                do_svd_truncation=True,
                initial_rank_ratio=merge_config.merge_rank_ratio,
                arch_name=merge_config.merge_backbone
            )
            models_to_merge_task_vectors.append(each_tv)
        
        print(f"merge coefficient: {merge_config.merge_coeff}")
        with torch.no_grad():

            merged_tv = models_to_merge_task_vectors[0] + \
                models_to_merge_task_vectors[1]

            for index in range(2, len(models_to_merge_task_vectors)):
                merged_tv = merged_tv + models_to_merge_task_vectors[index]

            merged_model = merged_tv.combine_with_pretrained_model(
                pretrained_model=avg_model,
                scaling_coefficient=merge_config.merge_coeff
            )

        return merged_model

    def cart_indexing(self, prev_origin, models_to_merge, exclude_param_names_regex, merge_config: DictConfig):
        assert merge_config.merge_rank_ratio is not None, f"current value of merge_rank_ratio is {merge_config.merge_rank_ratio}"
        assert merge_config.merge_coeff is not None, f"current value of merge_coeff is {merge_config.merge_coeff}"

        print("Construct Average model w/o head")
        avg_model: nn.Module = deepcopy(prev_origin)

        avg_origin = self.average_merging(
            models_to_merge=models_to_merge,
            exclude_param_names_regex=EXCLUDE_MAP[merge_config.merge_backbone]["avg"]
        )
        avg_model.load_state_dict(avg_origin, strict=False)
        print(
            f"Task Vector truncation rank ratio: {merge_config.merge_rank_ratio}")

        models_to_merge_task_vectors = []
        for each_model in tqdm(models_to_merge):
            _exclude_regex = get_exclude_regex(
                merge_config.merge_backbone, merge_config.decompose
            )
            each_tv = TaskVector(
                pretrained_model=avg_model,
                finetuned_model=each_model,
                exclude_param_names_regex=_exclude_regex,
                do_svd_truncation=True,
                initial_rank_ratio=merge_config.merge_rank_ratio,
                arch_name=merge_config.merge_backbone
            )
            models_to_merge_task_vectors.append(each_tv)

        print(f"merge coefficient: {merge_config.merge_coeff}")


        return {
            "model": avg_model,
            "tv_list": models_to_merge_task_vectors
        }

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

    def task_arithmetic(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
        assert isinstance(
            scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge,
                                                   exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        with torch.no_grad():
            merged_task_vector = models_to_merge_task_vectors[0] + \
                models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + \
                    models_to_merge_task_vectors[index]

            merged_params = merged_task_vector.combine_with_pretrained_model(
                pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

    def merging_models(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, trainers: list = None, scaling_coefficient: float = 1.0,
                       nums_fisher_examples: list = None, fisher_scaling_coefficients: list = None, normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6,
                       nums_regmean_examples: list = None, reduce_non_diagonal_ratio: float = 1.0, param_value_mask_rate: float = 0.8,
                       weight_format: str = "delta_weight", weight_mask_rates: list = None, use_weight_rescale: bool = True, mask_strategy: str = "random",
                       mask_apply_method: str = "average_merging", models_use_deepcopy: bool = False, merge_config: Optional[DictConfig] = None):
        merged_model = merged_model.to('cpu')
        if self.merging_method_name == "average_merging":
            merged_params = self.average_merging(
                models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex)
        elif self.merging_method_name == "task_arithmetic":
            merged_params = self.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                 scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "cart":
            assert merge_config is not None, "merge_config should not be None for CART_merging"
            merged_params = self.cart_merging(
                prev_origin=merged_model, 
                models_to_merge=models_to_merge,
                exclude_param_names_regex=exclude_param_names_regex,
                merge_config=merge_config
            )

            return merged_params

        else:
            raise NotImplementedError(
                f"unsupported for merging_method_name {self.merging_method_name}!")
        return merged_params

    def get_merged_model(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, trainers: list = None, scaling_coefficient: float = 1.0,
                         nums_fisher_examples: list = None, fisher_scaling_coefficients: list = None, normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6,
                         nums_regmean_examples: list = None, reduce_non_diagonal_ratio: float = 1.0, param_value_mask_rate: float = 0.8,
                         weight_format: str = "delta_weight", weight_mask_rates: list = None, use_weight_rescale: bool = True, mask_strategy: str = "random",
                         mask_apply_method: str = "average_merging", models_use_deepcopy: bool = False):
        if self.merging_method_name == "cart":
            merged_params = self.merging_models(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex, trainers=trainers,
                                                nums_fisher_examples=nums_fisher_examples, scaling_coefficient=scaling_coefficient, fisher_scaling_coefficients=fisher_scaling_coefficients,
                                                normalize_fisher_weight=normalize_fisher_weight, minimal_fisher_weight=minimal_fisher_weight,
                                                nums_regmean_examples=nums_regmean_examples, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio, param_value_mask_rate=param_value_mask_rate,
                                                weight_format=weight_format, weight_mask_rates=weight_mask_rates, use_weight_rescale=use_weight_rescale, mask_strategy=mask_strategy,
                                                mask_apply_method=mask_apply_method, models_use_deepcopy=models_use_deepcopy)
            self.copy_params_to_model(params=merged_params, model=merged_model)
            return merged_model

        elif self.merging_method_name == "ties_merging":
            merged_params = self.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                              param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
            self.copy_params_to_model(params=merged_params, model=merged_model)
            return merged_model
        return 

    def get_cart_merged_model(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, merge_config: DictConfig, models_use_deepcopy: bool = False) -> nn.Module:
        
        print(f"construct merged {merge_config.merge_method} parameters")
        if merge_config.merge_method == "cart":
            merged_statedict: dict[str, torch.Tensor] = self.merging_models(
                merged_model=merged_model,
                models_to_merge=models_to_merge,
                exclude_param_names_regex=exclude_param_names_regex,
                models_use_deepcopy=models_use_deepcopy,
                merge_config=merge_config
            )
            print("load state dict to merged model")

            _ret_model = copy.deepcopy(merged_model)
            self.copy_params_to_model(params=merged_statedict, model=_ret_model)

            return _ret_model
        return

