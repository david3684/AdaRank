import os

import torch
import pickle
from tqdm import tqdm
import math

import numpy as np

from src import MERGE_TYPE_LIST, MERGE_METHOD_LIST, MERGE_METHOD_STATIC_LIST, MERGE_METHOD_ADAPTIVE_LIST
from src.datasets.registry import tasks_8, tasks_20
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose

import gc
        
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model



def get_logits(inputs, classifier, dataset, args):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs, dataset, args)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def set_device(device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def softmax_entropy(x: torch.Tensor):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def check_config_sanity(config):
    # merge method check
    
    if config.merge_method == "individual" or "zeroshot":
        pass
    
    else:
        if config.merge_method not in MERGE_METHOD_LIST:
            raise ValueError(f"Invalid merge method: {config.merge_method}")
        
        if config.merge_type == "static":
            if config.merge_method not in MERGE_METHOD_STATIC_LIST:
                raise ValueError(f"Invalid merge method: {config.merge_method}, merge type: {config.merge_type}")
            
        if config.merge_type == "adaptive":
            if config.merge_method not in MERGE_METHOD_ADAPTIVE_LIST:
                raise ValueError(f"Invalid merge method: {config.merge_method}, merge type: {config.merge_type}")
    
    # valid tasks
    for task in config.tasks:
        if task not in tasks_20:
            raise ValueError(f"Invalid task: {task}")

# backward compatibility
def is_TA_mode(config: DictConfig, task_name: str)->bool:
    _ta_mode = config.get("TA_mode", False)
    is_target_task = task_name in tasks_8  
    
    if _ta_mode and is_target_task:
        print("currently load weight from TA authors")
        return True
    else:
        return False

def get_dir_dict(config: DictConfig, is_ta_mode: bool)->dict:
    
    if is_ta_mode:
        
        _weight_root =  ["/"] + config.weight_root.rstrip('/').split("/")
        weight_root = os.path.join(*_weight_root[:-1], _weight_root[-1] + "_TA")
        
        _save_dir = ["/"] + config.save.rstrip('/').split("/")
        save_dir = os.path.join(*_save_dir[:-1], _save_dir[-1] + "_TA")
    
    else:
        weight_root = config.weight_root
        save_dir = config.save
    
    return {
        "save": save_dir,
        "weight_root": weight_root
    }

def garbage_collect():
    gc.collect()
    torch.cuda.empty_cache()
    

def gpu_supports_bf16_via_pytorch():

    if not torch.cuda.is_available():
        return False
    
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        return False

    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
        return True
    except Exception as e:
        return False