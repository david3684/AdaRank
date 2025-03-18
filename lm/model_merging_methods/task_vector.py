import torch
import torch.nn as nn

from typing import Optional
from utils.utils import get_param_names_to_merge
from tqdm import tqdm

GPU_DEVICE: str = "cuda"

ARCH_NAME_LIST = [
    "roberta-base",
    "gpt2"
]

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


def truncated_svd(matrix: torch.Tensor, rank_ratio: float) -> torch.Tensor:
    _original_device = matrix.device
    matrix = matrix.to(GPU_DEVICE)
    if rank_ratio == 0.0:
        return torch.zeros_like(matrix).to(_original_device)

    elif rank_ratio == 1.0:
        return matrix.to(_original_device)

    else:
        U, s, V_t = torch.linalg.svd(matrix, full_matrices=False)
        rank_int: int = int(rank_ratio * s.shape[0])
        _recon = U[:, :rank_int] @ torch.diag(s[:rank_int]) @ V_t[:rank_int, :]
        return _recon.to(_original_device)

class TaskVector:
    def __init__(self, pretrained_model: Optional[nn.Module] = None, finetuned_model: Optional[nn.Module] = None, exclude_param_names_regex: Optional[list] = None, task_vector_param_dict: Optional[dict] = None, do_svd_truncation: bool = False, initial_rank_ratio: Optional[float] = None, arch_name: Optional[str] = None):
        if do_svd_truncation:
            assert arch_name is not None, f"arch_name should be provided when do_svd_truncation is True, currently {arch_name}"
            
            assert initial_rank_ratio is not None, f"initial_rank_ratio should be provided when do_svd_truncation is True, currently {initial_rank_ratio}"

        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name,
                                     param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name,
                                    param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(
                pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in tqdm(param_names_to_merge):
                    if do_svd_truncation:
                        if is_mat_params(arch_name, param_name):
                            self.task_vector_param_dict[param_name] = truncated_svd(
                                matrix=finetuned_param_dict[param_name] - pretrained_param_dict[param_name], rank_ratio=initial_rank_ratio)
                        else:
                            self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - \
                                pretrained_param_dict[param_name]
                    else:
                        self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - \
                            pretrained_param_dict[param_name]

    def __add__(self, other):
        assert isinstance(
            other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(
                ), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + \
                    other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0)->dict[str, torch.Tensor]:
        pretrained_param_dict = {param_name: param_value for param_name,
                                 param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + \
                    scaling_coefficient * \
                    self.task_vector_param_dict[param_name]

        return merged_params
