import os
import torch.utils
from omegaconf import OmegaConf, DictConfig
import argparse
from functools import partial
import time
import torch
print("Visible GPUs:", torch.cuda.device_count())
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import torch.nn as nn
import torch.optim as optim
from model_merging_methods.adarank_roberta import AdaRankModule
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from utils.load_config import cache_dir, weight_dir
from model_merging_methods.task_vector import *

dataset_model_learning_rate_mapping_dict = {
    "cola_bert-base-uncased": 5e-5,
    "sst2_bert-base-uncased": 1e-5,
    "mrpc_bert-base-uncased": 5e-5,
    "qqp_bert-base-uncased": 1e-5,
    "mnli_bert-base-uncased": 1e-5,
    "qnli_bert-base-uncased": 1e-5,
    "rte_bert-base-uncased": 1e-5,
    "cola_roberta-base": 1e-5,
    "sst2_roberta-base": 1e-5,
    "mrpc_roberta-base": 5e-5,
    "qqp_roberta-base": 1e-5,
    "mnli_roberta-base": 1e-5,
    "qnli_roberta-base": 1e-5,
    "rte_roberta-base": 1e-5
}

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def run_block(args, merge_config):
    args.language_model_name = merge_config.merge_backbone
    args.merging_method_name = merge_config.merge_method
    args.batch_size = merge_config.eval_batch_size

    args.dataset_names = ["cola", "sst2", "mrpc",
                        "qqp", "mnli", "qnli", "rte"]
    assert all([dataset_name in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"] for dataset_name in args.dataset_names]), \
        'name in dataset_names must be contained in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]!'
    load_model_paths = []
    for dataset_name in args.dataset_names:
        learning_rate = dataset_model_learning_rate_mapping_dict[
            f"{dataset_name}_{args.language_model_name}"]
        load_model_paths.append(
            f"{weight_dir}/roberta/{dataset_name}/{args.language_model_name}_lr{learning_rate}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name), use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    models_to_merge, tta_dataloaders, trainers = [], [], []
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             train_split_ratio_for_val=0.1,
                                                                                             max_seq_length=128)
        training_args = TrainingArguments(
            output_dir=load_model_path,                        
            per_device_train_batch_size=merge_config.tta_batch_size,
            per_device_eval_batch_size=args.batch_size, 
            report_to="none"
        )

        assert os.path.exists(os.path.join(
            training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
        model_to_merge = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=training_args.output_dir,
            num_labels=num_labels).to(args.device)
        models_to_merge.append(model_to_merge)

        trainer = CustomizedTrainer(
            model=model_to_merge,
            args=training_args,
            train_dataset=test_dataset,
            eval_dataset=test_dataset,
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),
            tokenizer=tokenizer,
        )
        trainers.append(trainer)
        tta_dataloader = trainer.get_train_dataloader()
        tta_dataloaders.append(tta_dataloader)
    
    tta_dataloader_iters = [iter(dl) for dl in tta_dataloaders]
    tqdm_iterator = tqdm(range(merge_config.tta_steps), desc="TTA steps", dynamic_ncols=True)
    
    adaptive_model = AdaRankModule(
        config=merge_config, models_to_merge=models_to_merge)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, adaptive_model.parameters()),
        lr=merge_config.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    pseudo_idx = str(time.time())
    
    print(f"********** Run starts. **********")

    merge_config.index = pseudo_idx

    
            
    losses = 0.0
    losses_log = {}
    ent = []
    for step in tqdm_iterator:
        if not merge_config.skip_first_eval and (step == 0):
            model = adaptive_model.get_model()
            print(f"Evaluation in step 0")
            for idx, (dataset_name, trainer) in enumerate(zip(args.dataset_names, trainers)):
                model.classifier = models_to_merge[idx].classifier
                merged_model_evaluator = CustomizedTrainer(
                    model=model,
                    args=trainer.args,
                    eval_dataset=trainer.eval_dataset,
                    compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),
                    tokenizer=tokenizer,
                )
                print(
                    f"perform model merging method {args.merging_method_name}:")
                print(f"get performance...")
                test_metrics = merged_model_evaluator.evaluate()
                test_metrics = {k: float(f"{v:.4f}") if isinstance(
                        v, float) else v for k, v in test_metrics.items()}
                test_metric_method = f"eval_{glue_data_metrics_map[dataset_name]}"
                
                if test_metric_method != "eval_accuracy" and "eval_accuracy" in test_metrics:
                    print(f"Accuracy/{dataset_name}/eval_accuracy: {test_metrics['eval_accuracy']}")
                else:
                    print(f"Accuracy/{dataset_name}/{test_metric_method}: {test_metrics[test_metric_method]}")


        
        adaptive_model.reset_merged_state()
        optimizer.zero_grad()
        losses = 0.0
        first = True
        for idx, dataset_name in enumerate(args.dataset_names):
            try:
                batch = next(tta_dataloader_iters[idx])
            except StopIteration:
                tta_dataloader_iters[idx] = iter(tta_dataloaders[idx])
                batch = next(tta_dataloader_iters[idx])

            batch = {k: v.to(merge_config.device) for k, v in batch.items()}

            loss, _ = compute_loss(adaptive_model, batch, idx)

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
            adaptive_model.reset_merged_state()
            adaptive_model._merge_weights()
            adaptive_model.copy_params_to_model(adaptive_model._merged_state_dict, adaptive_model.base_model)
            model = adaptive_model.base_model
            for idx, (dataset_name, trainer) in enumerate(zip(args.dataset_names, trainers)):
                model.classifier = models_to_merge[idx].classifier
                merged_model_evaluator = CustomizedTrainer(
                    model=model,
                    args=trainer.args,
                    eval_dataset=trainer.eval_dataset,
                    compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),
                    tokenizer=tokenizer,
                )
                print(
                    f"perform model merging method {args.merging_method_name}:")
                print(f"get performance...")
                test_metrics = merged_model_evaluator.evaluate()
                test_metrics = {k: float(f"{v:.4f}") if isinstance(
                        v, float) else v for k, v in test_metrics.items()}
                test_metric_method = f"eval_{glue_data_metrics_map[dataset_name]}"
                
                if test_metric_method != "eval_accuracy" and "eval_accuracy" in test_metrics:
                    print(f"Accuracy/{dataset_name}/eval_accuracy: {test_metrics['eval_accuracy']}")
                else:
                    print(f"Accuracy/{dataset_name}/{test_metric_method}: {test_metrics[test_metric_method]}")

def softmax_entropy(x: torch.Tensor):
    return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)

def compute_loss(model: nn.Module, inputs: dict, idx: int):
    if "labels" in inputs:
        inputs.pop("labels")
    inputs["dataset_ids"] = idx
    outputs = model(**inputs) 
    logits = outputs["logits"]

    loss = softmax_entropy(logits).mean(dim=0)

    return loss, outputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Interface for merging roberta models on glue")
    parser.add_argument(
        "--exp_config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0,
                        help="number of gpu to use")
    parser.add_argument("--run_parallel", action="store_true")

    args = parser.parse_args()
    args.device = f"cuda"
    
    merge_config: DictConfig = OmegaConf.load(args.exp_config)
    run_block(args, merge_config)


