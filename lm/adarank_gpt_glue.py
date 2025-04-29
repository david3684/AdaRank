import os
import torch
print("Visible GPUs:", torch.cuda.device_count())
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from transformers import (
    GPT2ForSequenceClassification,
    default_data_collator,
    AutoConfig
)
from utils.load_config import cache_dir
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import torch.optim as optim
from model_merging_methods.adarank_gpt2 import AdaRankModule
import argparse
from torch.utils.data import DataLoader
import time
from functools import partial
from torchmetrics import Accuracy, MatthewsCorrCoef
import torch
from transformers import GPT2TokenizerFast
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

num_labels = {
    'cola': 2,
    'sst2': 2,
    'mrpc': 2,
    'qqp': 2,
    'mnli': 3,
    'qnli': 2,
    'rte': 2
}
dataset_names = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]

MODEL_NAME = "gpt2"
GPT_CLSs = "avg"
CPU_DEVICE: str = "cpu"
WEIGHT_DIR: str = "/data1/common_datasets/merge_vision_weight/checkpoints/lm/gpt2"


def run_block(args: argparse.Namespace, merge_config: DictConfig):
    args.dataset_names = dataset_names
    
    args.language_model_name = merge_config.merge_backbone
    args.merging_method_name = merge_config.merge_method
    args.batch_size = merge_config.eval_batch_size
    
    
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path=os.path.join(WEIGHT_DIR, "gpt2"))

    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token


    models = []

    for dataset_name in tqdm(dataset_names, desc="loading finetuned weight"):
        args.dataset_name = dataset_name
        load_model_path = os.path.join(WEIGHT_DIR, f"gpt2_{dataset_name}")
        finetuned_model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=load_model_path).to(merge_config.device)
        models.append(finetuned_model)

    _pseudo_idx = str(time.time())
    merge_config.index = _pseudo_idx
    
    print(f"********** Run starts. **********")
    print(f"configuration is {args}")
    
    adaptive_model = AdaRankModule(
        config=merge_config, models_to_merge=models)
    tqdm_iterator = tqdm(range(merge_config.tta_steps), desc="TTA steps", dynamic_ncols=True)
    pretrained_model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=os.path.join(WEIGHT_DIR, "gpt2")).to(args.device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, adaptive_model.parameters()),
        lr=merge_config.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    losses = 0.0
    losses_log = {}
    tta_dataloader = []
    for idx, dataset_name in enumerate(args.dataset_names):
        glue = TokenizedGLUE(tokenizer)
        ds = glue.load_dataset(dataset_name)
        try:
            ds_val = ds['validation']
        except:
            ds_val = ds['validation_mismatched']
        loader = DataLoader(
            ds_val,
            collate_fn=default_data_collator,
            batch_size=merge_config.tta_batch_size,
            num_workers=1,
            shuffle=True,
        )
        tta_dataloader.append(loader)
    tta_dataloader_iters = [iter(dl) for dl in tta_dataloader]

    for step in tqdm_iterator:
        if not merge_config.skip_first_eval and (step == 0):
            model = adaptive_model.get_model()
            for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models)):
                model.config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=os.path.join(WEIGHT_DIR, f"gpt2_{dataset_name}"))
                
                model.score = model_to_merge.score
                model.to(args.device)
                glue = TokenizedGLUE(tokenizer)
                ds = glue.load_dataset(dataset_name)

                try:
                    ds_val = ds['validation']
                except:
                    ds_val = ds['validation_mismatched']
                if dataset_name != "cola":
                    with torch.no_grad():
                        accuracy = Accuracy("multiclass", num_classes=num_labels[
                            dataset_name])
                        loader = DataLoader(
                            ds_val,
                            collate_fn=default_data_collator,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=True,
                        )
                        for batch in (
                                pbar := tqdm(
                                    loader, desc="Evaluating", leave=False, dynamic_ncols=True
                                )
                        ):
                            input_ids = batch["input_ids"].to(args.device)
                            attention_mask = batch["attention_mask"].to(args.device)
                            labels = batch["labels"].to(args.device)

                            outputs = model(
                                input_ids, attention_mask=attention_mask)
                            logits = outputs.logits
                            acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                            
                            pbar.set_postfix({"accuracy": acc.item()})
                        acc = accuracy.compute().item()
                        print(f"Accuracy/{dataset_name}: {acc}")
                else:
                    with torch.no_grad():
                        mcc = MatthewsCorrCoef("multiclass", num_classes=num_labels[
                            dataset_name])
                        loader = DataLoader(
                            ds_val,
                            collate_fn=default_data_collator,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=True,
                        )
                        for batch in (
                                pbar := tqdm(
                                    loader, desc="Evaluating", leave=False, dynamic_ncols=True
                                )
                        ):
                            input_ids = batch["input_ids"].to(args.device)
                            attention_mask = batch["attention_mask"].to(args.device)
                            labels = batch["labels"].to(args.device)

                            outputs = model(
                                input_ids, attention_mask=attention_mask)
                            logits = outputs.logits
                            acc = mcc(logits.detach().cpu(), labels.detach().cpu())
                            
                            pbar.set_postfix({"mcc": acc.item()})
                        acc = mcc.compute().item()
                        print(f"MCC/{dataset_name}: {acc}")
        
        adaptive_model.reset_merged_state()
        optimizer.zero_grad()
        losses = 0.0
        first = True
        for idx, dataset_name in enumerate(args.dataset_names):
            try:
                batch = next(tta_dataloader_iters[idx])
            except StopIteration:
                tta_dataloader_iters[idx] = iter(tta_dataloader[idx])
                batch = next(tta_dataloader_iters[idx])

            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            
            
            loss, _ = compute_loss(adaptive_model, input_ids, attention_mask, idx)
            losses_log[f"Loss/Per-sample/{dataset_name}"] = loss.item()
            
            if first:
                losses = loss
                first = False
            else:
                losses += loss
            
            losses_log[f"Loss/Total/{dataset_name}"] = loss.item()

        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % merge_config.tta_eval_interval == 0:
            if step == 0:
                continue
            model = adaptive_model.get_model()
            for idx, (dataset_name, model_to_merge) in enumerate(zip(args.dataset_names, models)):
                model.config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=os.path.join(WEIGHT_DIR, f"gpt2_{dataset_name}"))
                
                model.score = model_to_merge.score
                model.to(args.device)
                glue = TokenizedGLUE(tokenizer)
                ds = glue.load_dataset(dataset_name)

                try:
                    ds_val = ds['validation']
                except:
                    ds_val = ds['validation_mismatched']
                if dataset_name != "cola":
                    with torch.no_grad():
                        accuracy = Accuracy("multiclass", num_classes=num_labels[
                            dataset_name])
                        loader = DataLoader(
                            ds_val,
                            collate_fn=default_data_collator,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=True,
                        )
                        for batch in (
                                pbar := tqdm(
                                    loader, desc="Evaluating", leave=False, dynamic_ncols=True
                                )
                        ):
                            input_ids = batch["input_ids"].to(args.device)
                            attention_mask = batch["attention_mask"].to(args.device)
                            labels = batch["labels"].to(args.device)

                            outputs = model(
                                input_ids, attention_mask=attention_mask)
                            logits = outputs.logits
                            acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
                            
                            pbar.set_postfix({"accuracy": acc.item()})
                        acc = accuracy.compute().item()
                        print(f"Accuracy/{dataset_name}: {acc}")
                else:
                    with torch.no_grad():
                        mcc = MatthewsCorrCoef("multiclass", num_classes=num_labels[
                            dataset_name])
                        loader = DataLoader(
                            ds_val,
                            collate_fn=default_data_collator,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=True,
                        )
                        for batch in (
                                pbar := tqdm(
                                    loader, desc="Evaluating", leave=False, dynamic_ncols=True
                                )
                        ):
                            input_ids = batch["input_ids"].to(args.device)
                            attention_mask = batch["attention_mask"].to(args.device)
                            labels = batch["labels"].to(args.device)

                            outputs = model(
                                input_ids, attention_mask=attention_mask)
                            logits = outputs.logits
                            acc = mcc(logits.detach().cpu(), labels.detach().cpu())
                            
                            pbar.set_postfix({"mcc": acc.item()})
                        acc = mcc.compute().item()
                        print(f"MCC/{dataset_name}: {acc}")


def softmax_entropy(x: torch.Tensor):
    return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)

def compute_loss(model: nn.Module, input_ids, attention_mask, idx: int):
    
    outputs = model(idx, input_ids, attention_mask = attention_mask)
    logits = outputs.logits

    loss = softmax_entropy(logits).mean(dim=0)

    return loss, outputs

def mrpc_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples['sentence1'], 
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs

def mnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def cola_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qnli_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        examples["sentence"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def qqp_tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question1"],
        examples["question2"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


class TokenizedGLUE:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def load_dataset(
        self, name
    ):
        glue_dataset_loaders = {
            "mrpc": self.load_mrpc_dataset,
            "mnli": self.load_mnli_dataset,
            "cola": self.load_cola_dataset,
            "sst2": self.load_sst2_dataset,
            "qnli": self.load_qnli_dataset,
            "qqp": self.load_qqp_dataset,
            "rte": self.load_rte_dataset,
            # "wnli": load_wnli_dataset,
        }
        return glue_dataset_loaders[name]()

    def load_mrpc_dataset(self):
        task_subdir = os.path.join(cache_dir, "mrpc")
        dataset = load_from_disk(task_subdir)
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['sentence1', 'sentence2'],
        )
        return dataset

    def load_rte_dataset(self):
        task_subdir = os.path.join(cache_dir, "rte")
        dataset = load_from_disk(task_subdir)
        dataset = dataset.map(
            # RTE has the same format as MRPC
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    # not used
    def load_wnli_dataset(self):
        dataset = load_dataset("glue", "wnli", cache_dir=cache_dir)
        dataset = dataset.map(
            partial(mrpc_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    def load_qqp_dataset(self):
        task_subdir = os.path.join(cache_dir, "qqp")
        dataset = load_from_disk(task_subdir)
        dataset = dataset.map(
            partial(qqp_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=['question1', 'question2'],
        )
        return dataset

    def load_mnli_dataset(self):
        task_subdir = os.path.join(cache_dir, "mnli")
        dataset = load_from_disk(task_subdir)
        dataset = dataset.map(
            partial(mnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        )
        return dataset

    def load_cola_dataset(self):
        task_subdir = os.path.join(cache_dir, "cola")
        dataset = load_from_disk(task_subdir)
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset

    def load_sst2_dataset(self):
        task_subdir = os.path.join(cache_dir, "sst2")
        dataset = load_from_disk(task_subdir)
        print(dataset.column_names)
        dataset = dataset.map(
            partial(cola_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["sentence"],
        )
        return dataset

    def load_qnli_dataset(self):
        task_subdir = os.path.join(cache_dir, "qnli")
        dataset = load_from_disk(task_subdir)
        dataset = dataset.map(
            partial(qnli_tokenize_function, tokenizer=self.tokenizer),
            batched=True,
            remove_columns=["question", "sentence"],
        )
        return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Interface for inference PLMs on glue")

    parser.add_argument("--gpu", type=int, default=0,
                        help="number of gpu to use")
    parser.add_argument("--run_parallel", action="store_true",)
    parser.add_argument(
        "--exp_config", type=str, required=True)
    
    args = parser.parse_args()
    
    merge_config = OmegaConf.load(args.exp_config)
    args.device = merge_config.device
    run_block(args, merge_config=merge_config)
