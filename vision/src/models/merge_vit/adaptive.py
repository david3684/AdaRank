import torch
import torch.nn as nn
import functools
from copy import deepcopy
from torch.nn.utils import stateless
from src.models.heads import get_classification_head

from tqdm import tqdm

CPU_DEVICE = "cpu"


class AdaMergingModule(nn.Module):
    '''
    Module for Adaptive Merging
    '''

    def __init__(self, config, zero_shot_encoder, task_vectors):
        super(AdaMergingModule, self).__init__()
        self.config = config
        # Average(CART) or pretrained model(TA)
        self.origin = zero_shot_encoder
        self.exam_datasets = config.tasks
        self.clamp_weights = getattr(self.config, "clamp_weights", False)
        self.extend_clamp = getattr(self.config, "extend_clamp", False)
        self.soft_mask = getattr(self.config, "soft_mask", False)

        self.normalized_merging_weights = getattr(
            self.config, "normalized_merging_weights", False)
        self.device = config.device

        if config.initial_rank_ratio == 0:
            config.initial_rank_ratio = 1 / len(task_vectors)

        self.svd_list = self._svd_vanilla(task_vectors)

        rlambdas = torch.ones(len(task_vectors), len(
            self.origin.state_dict())) * config.prior
        self.merge_weight = torch.nn.Parameter(rlambdas)

        self.merge_mask = nn.ParameterList(self._mask_init(task_vectors))
        self.mask_temp = config.mask_temp

        
        print("Module initialized with initial rank ratio:", config.initial_rank_ratio)
        print("Module initialized with merge method:", config.merge_method)

        self.classifier_names = []
        for dataset_name in self.exam_datasets:
            classification_head = get_classification_head(
                self.config, dataset_name)
            layer_name = f"classifier_{dataset_name}"
            self.add_module(layer_name, classification_head.to(self.device))
            self.classifier_names.append(layer_name)

        self._overall_requires_grad()
        self._merged_state_dict = None

    def straight_through_mask(self, mat_svd, mask):
        U, s, V_T = mat_svd
        s_masked = mask * s + (((mask > 0.5).float() - mask) * s).detach()
        return (U * s_masked) @ V_T

    def soft_mask_func(self, mat_svd, mask):
        U, s, V_T = mat_svd
        s_masked = mask * s
        return (U * s_masked) @ V_T

    def _overall_requires_grad(self):
        """
        Only merge_weight and merge_mask require gradients.
        """
        for name, param in self.named_parameters():
            param.requires_grad = False
            if "merge_weight" in name or "merge_mask" in name:
                param.requires_grad = True

    def forward(self, x, dataset_name):
        if self._merged_state_dict is None:
            self._merge_weights()
        features = self.forward_model(x, dataset=None, args=None)
        layer_name = f"classifier_{dataset_name}"
        classification_head = getattr(self, layer_name)
        out = classification_head(features)
        return out

    def forward_model(self, inp, dataset=None, args=None):
        partial_functional_call = functools.partial(
            stateless.functional_call,
            self.origin,
            self._merged_state_dict,
        )
        return partial_functional_call((inp, dataset, args))

    def _merge_weights(self):
        origin_state = {key: value.detach().clone()
                        for key, value in self.origin.state_dict().items()}
        state_dict = origin_state

        if self.clamp_weights:
            if self.extend_clamp:
                layer_wise_weight = self.merge_weight.clamp(-0.5, 2)
            else:
                layer_wise_weight = self.merge_weight.clamp(0, 1)
        else:
            layer_wise_weight = self.merge_weight
        if self.normalized_merging_weights:
            layer_wise_weight = layer_wise_weight.softmax(dim=0)

        for task_idx, (weight, each_task_vector) in enumerate(zip(layer_wise_weight, self.svd_list)):
            for w, m, (key, value) in zip(weight, self.merge_mask, each_task_vector.items()):
                if ('attn' in key or 'mlp' in key) and not ('ln' in key or 'bias' in key):
                    _val = value

                    if self.soft_mask:
                        task_vector = self.soft_mask_func(
                            _val, (m[task_idx] / self.mask_temp).sigmoid())
                    else:
                        task_vector = self.straight_through_mask(
                            _val, (m[task_idx] / self.mask_temp).sigmoid())
                else:
                    task_vector = value
                state_dict[key].add_(w * task_vector)
        self._merged_state_dict = state_dict

    def get_image_encoder(self):
        if self._merged_state_dict is None:
            self._merge_weights()
        clone_model = deepcopy(self.origin)
        clone_model.load_state_dict(self._merged_state_dict)
        return clone_model

    def get_classification_head(self, dataset_name):
        return getattr(self, f"classifier_{dataset_name}")

    def reset_merged_state(self):
        self._merged_state_dict = None

    def _get_origin(self, task_vectors):

        if self.config.merge_method == "CART":
            state_dict = deepcopy(self.origin.state_dict())
            coeff = 1.0 / len(self.exam_datasets)
            processed_tvec = sum(task_vectors)
            for key in state_dict.keys():
                state_dict[key] = state_dict[key] + \
                    coeff * processed_tvec.vector[key]
            return state_dict
        else:
            return deepcopy(self.origin.state_dict())

    def _svd_vanilla(self, task_vectors):
        svd_list = []
        for idx, task_vector in tqdm(enumerate(task_vectors), total=len(task_vectors), desc="Decompose task vectors"):
            svd_vector = {}
            for key, value in tqdm(task_vector.vector.items(), desc=f"Decompose task {idx}"):
                if ('attn' in key or 'mlp' in key) and not ('ln' in key or 'bias' in key):
                    _val = value
                    ret = torch.linalg.svd(_val, full_matrices=False)
                    svd_vector[key] = ret
                else:
                    svd_vector[key] = value.to(self.device)
            svd_list.append(svd_vector)
        return svd_list

    def _svd_with_truncation_ratio(self, task_vectors, rank_ratio):
        svd_list = []
        for task_vector in task_vectors:
            svd_vector = {}
            for key, value in task_vector.vector.items():
                if ('attn' in key or 'mlp' in key) and not ('ln' in key or 'bias' in key):
                    _val = value.to(self.device)
                    U, s, V_T = torch.linalg.svd(_val, full_matrices=False)
                    full_dim = min(_val.shape[0], _val.shape[1])
                    truncated_dim = max(1, int(rank_ratio * full_dim))
                    U_truncated = U[:, :truncated_dim]
                    s_truncated = s[:truncated_dim]
                    V_T_truncated = V_T[:truncated_dim, :]
                    svd_vector[key] = (U_truncated, s_truncated, V_T_truncated)
                else:
                    svd_vector[key] = value.to(self.device)
            svd_list.append(svd_vector)
        return svd_list

    def _mask_init(self, task_vectors):
        self.svd_keys = []
        merge_mask = []
        for key, value in self.origin.state_dict().items():
            if ('attn' in key or 'mlp' in key) and not ('ln' in key or 'bias' in key):
                self.svd_keys.append(key)
                full_dim = min(value.shape[0], value.shape[1])
                
                dim = full_dim
                if self.soft_mask:
                    mask = 2.0 * \
                        torch.ones(len(task_vectors), dim,
                                    dtype=torch.float32)
                else:
                    mask = 0.1 * \
                        torch.ones(len(task_vectors), dim,
                                    dtype=torch.float32)
                if (self.config.initial_rank_ratio < 1.0):
                    preserved_dim = int(
                        dim * self.config.initial_rank_ratio)
                    if self.soft_mask:
                        mask[:, preserved_dim:] = 0.0
                    else:
                        mask[:, preserved_dim:] = -0.1
                merge_mask.append(torch.nn.Parameter(mask, requires_grad=True))
            else:
                merge_mask.append(torch.nn.Parameter(torch.ones(1)))
        return merge_mask
