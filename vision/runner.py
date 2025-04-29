import os
import glob
import logging
import ray
from omegaconf import OmegaConf, DictConfig
import torch
from torch import optim

from bitsandbytes.optim import Adam8bit
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from src.datasets.common import maybe_dictionarize
from src.utils import softmax_entropy
from src.utils import check_config_sanity, set_device, garbage_collect
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader_shuffle, get_dataloader, maybe_dictionarize
from src.eval import eval_single_dataset
from src.models.heads import get_classification_head
from src.models.modeling import ImageEncoder
from src.models.task_vectors import TaskVector
from src.models.merge_vit.static import StaticMergeModule
from src.models.merge_vit.adaptive import AdaMergingModule
from src.logger import ExperimentResult

CPU_DEVICE = "cpu"


def average_finetuned_weights(config, model_list, tasks):
    common_keys = list(model_list[tasks[0]].keys())
    avg_state_dict = {key: torch.zeros_like(model_list[tasks[0]][key])
                        for key in common_keys}
    for key in common_keys:
        for finetuned_model in model_list.values():
            avg_state_dict[key] += finetuned_model[key]
        avg_state_dict[key] /= len(model_list)
    return avg_state_dict

def get_dir_dict(config: DictConfig, is_ta_mode: bool) -> dict:
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
    
def is_TA_mode(config: DictConfig, task_name: str) -> bool:
    _ta_mode = config.get("TA_MODE", False)
    tasks_8 = ['Cars','DTD','EuroSAT','GTSRB','MNIST','RESISC45','SUN397','SVHN']
    is_target_task = task_name in tasks_8
    
    if _ta_mode and is_target_task:
        print("currently load weight from TA authors")
        return True
    else:
        return False
    
def load_model(config, device, get_raw_weight=False):
    """
    load origin and task vectors for CLIP-ViT
    """
    logging.info("Loading models...")
    model_list = {}
    zero_shot_encoder = ImageEncoder(args=config, keep_lang=False).to(device)
    for name in config.tasks:

        TA_mode = is_TA_mode(config, name)
        dir_ret = get_dir_dict(config, TA_mode) 
        if TA_mode:
            path_name = name
            finetuned_model_path = os.path.join(
                dir_ret["weight_root"], path_name, "finetuned.pt"
            )
        else:
            path_name = name + "Val"
            finetuned_model_path = os.path.join(
                dir_ret["weight_root"], path_name, "nonlinear_finetuned.pt"
            )

        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(
                f"Model file not found: {finetuned_model_path}")
        model = torch.load(finetuned_model_path, map_location=device)
        model_list[name] = model

    if get_raw_weight:
        return {"origin": zero_shot_encoder, "task_vectors": model_list}

    logging.info("Constructing task vectors...")
    task_vectors = []

    # move origin to average
    if config.merge_method == "CART":
        avg_state_dict = average_finetuned_weights(
            config, model_list, config.tasks)
        zero_shot_encoder.load_state_dict(avg_state_dict)

    for task in config.tasks:
        finetuned_state_dict = model_list[task]
        tv = TaskVector(config, zero_shot_encoder.state_dict(),
                        finetuned_state_dict, task=task).to(device)
        task_vectors.append(tv)

    return {"origin": zero_shot_encoder, "task_vectors": task_vectors}


def eval_individual_checkpoints(config):
    ret = load_model(config, config.device, get_raw_weight=True)
    zero_shot_encoder = ret["origin"]
    model_list = ret["task_vectors"]

    for task in config.tasks:
        logging.info("Evaluating task: %s", task)
        image_encoder = zero_shot_encoder
        load_result = image_encoder.load_state_dict(model_list[task])
        logging.info("Load result: %s", load_result)
        classification_head = get_classification_head(config, task)
        metrics = eval_single_dataset(
            image_encoder=image_encoder,
            classification_head=classification_head,
            dataset_name=task,
            args=config,
        )

    return None

def eval_zeroshot(config):
    device = config.device
    zero_shot_encoder = ImageEncoder(args=config, keep_lang=False).to(device)

    test_log = ExperimentResult(
        merge_type=config.merge_type,
        method=config.merge_method,
        num_tasks=len(config.tasks),
        index=None,  
        exp_config=config,
    )

    for task in config.tasks:
        logging.info("Evaluating task: %s", task)
        image_encoder = zero_shot_encoder
        classification_head = get_classification_head(config, task)
        metrics = eval_single_dataset(
            image_encoder=image_encoder,
            classification_head=classification_head,
            dataset_name=task,
            args=config,
        )
        test_log.add_score(task, metrics.get("top1", 0.0))

    return test_log

def static_merge_and_eval(config):
    logging.info("Evaluating merge method: %s", config.merge_method)
    ret = load_model(config, CPU_DEVICE)
    mtl_model = StaticMergeModule(
        config=config,
        zero_shot_encoder=ret["origin"],
        task_vectors=ret["task_vectors"],
    )

    image_encoder = mtl_model.get_image_encoder(config.merge_method)
    test_log = ExperimentResult(
        method=config.merge_method,
        merge_type=config.merge_type,
        num_tasks=len(config.tasks),
        index=None,
        exp_config=config,
    )
    for task in config.tasks:
        logging.info("Evaluating task: %s", task)
        classification_head = get_classification_head(config, task)
        metrics = eval_single_dataset(
            image_encoder=image_encoder.to(config.device),
            classification_head=classification_head.to(config.device),
            dataset_name=task,
            args=config,
        )
        test_log.add_score(task, metrics.get("top1", 0.0))

    return test_log


def adaptive_merge_and_eval(config, device):        
    _device = device

    ret = load_model(config, _device)
    adaptive_model = AdaMergingModule(
        config=config,
        zero_shot_encoder=ret["origin"],
        task_vectors=ret["task_vectors"],
    ).to(device)
    
    if config.use_8bit_adam:
        optimizer = Adam8bit(
            filter(lambda p: p.requires_grad, adaptive_model.parameters()),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, adaptive_model.parameters()),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

    logging.info("Preparing DataLoaders...")
    tta_dataloaders = []
    for task_name in config.tasks:
        logging.info("Loading dataset for task: %s", task_name)
        dataset = get_dataset(
            dataset_name=task_name,
            preprocess=ret["origin"].val_preprocess,
            location=config.data_location,
            batch_size=config.tta_batch_size,
            persistent_workers=False,
            num_workers=4,
            subset_ratio=config.test_data_ratio
        )
        if config.test_data_ratio != 1:
            dataloader = get_dataloader(is_train=False)
        dataloader = get_dataloader_shuffle(dataset)
        tta_dataloaders.append(dataloader)

    del ret
    garbage_collect()

    tta_dataloader_iters = [iter(dl) for dl in tta_dataloaders]
    tqdm_iterator = tqdm(range(config.tta_steps), desc="TTA steps", dynamic_ncols=True)

    if config.half_precision:
        scaler = GradScaler()
        
    best_acc = -1.0
    best_task_accuracies = None
    avg_acc, task_accuracies = get_results(adaptive_model, config)
    logging.info("Initial evaluation - Average Accuracy: %.4f", avg_acc)
    best_acc = avg_acc
    best_task_accuracies = task_accuracies
        
    for step in tqdm_iterator:
        adaptive_model.reset_merged_state()
        if config.half_precision:
            optimizer.zero_grad()
            loss_dict = {}
            losses = 0.0
            losses_log = {}
            with autocast(dtype = torch.bfloat16)
                for idx, task in enumerate(config.tasks):
                    try:
                        data = next(tta_dataloader_iters[idx])
                    except StopIteration:
                        tta_dataloader_iters[idx] = iter(tta_dataloaders[idx])
                        data = next(tta_dataloader_iters[idx])
                    data = maybe_dictionarize(data)
                    x = data["images"].to(device)
                    outputs = adaptive_model(x, task)
                    
                    per_sample_loss = softmax_entropy(outputs).mean(dim=0) 
                    loss = per_sample_loss
                    
                    loss_dict[task] = loss.item()
                    losses += loss
                    losses_log[f"Loss/Per-sample/{task}"] = per_sample_loss.item()
                    losses_log[f"Loss/Total/{task}"] = loss.item()            
            
                avg_loss = losses.item() / len(config.tasks)
                tqdm_iterator.set_description(
                    f"TTA step: {step}, {loss_dict}, overall_loss: {avg_loss}"
                )
                
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            
        else:
            optimizer.zero_grad()
            loss_dict = {}
            losses = 0.0
            losses_log = {}
            for idx, task in enumerate(config.tasks):
                try:
                    data = next(tta_dataloader_iters[idx])
                except StopIteration:
                    tta_dataloader_iters[idx] = iter(tta_dataloaders[idx])
                    data = next(tta_dataloader_iters[idx])
                data = maybe_dictionarize(data)
                x = data["images"].to(device)
                outputs = adaptive_model(x, task)
                
                per_sample_loss = softmax_entropy(outputs).mean(dim=0) 
                loss = per_sample_loss
                
                loss_dict[task] = loss.item()
                losses += loss
                losses_log[f"Loss/Per-sample/{task}"] = per_sample_loss.item()
                losses_log[f"Loss/Total/{task}"] = loss.item()            
        
            avg_loss = losses.item() / len(config.tasks)
            tqdm_iterator.set_description(
                f"TTA step: {step}, {loss_dict}, overall_loss: {avg_loss}"
            )
            
            losses.backward()
            optimizer.step()
            
        if step % config.tta_eval_interval == 0:
            if step == 0:
                continue
            else:
                avg_acc, task_accuracies = get_results(adaptive_model, config)
                logging.info("TTA step: %s, avg_acc: %s", step, avg_acc)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_task_accuracies = task_accuracies

    if best_task_accuracies is None:
        avg_acc, best_task_accuracies = get_results(adaptive_model, config)
        best_acc = avg_acc

    return None


def get_results(adamerging_mtl_model, config):
    adamerging_mtl_model.reset_merged_state()
    print("Evaluating Start")
    total_acc = 0.0
    task_accuracies = []

    for task in config.tasks:
        print("Evaluating task: ", task)
        image_encoder = adamerging_mtl_model.get_image_encoder()
        classification_head = adamerging_mtl_model.get_classification_head(task)
        metrics = eval_single_dataset(
            image_encoder=image_encoder,
            classification_head=classification_head,
            dataset_name=task,
            args=config,
        )
        acc = metrics.get("top1", 0.0)
        total_acc += acc
        task_accuracies.append(acc)

    avg_acc = total_acc / len(config.tasks)
    return avg_acc, task_accuracies


def run_experiment(cfg: DictConfig):
    check_config_sanity(cfg)
    cfg.device = set_device(cfg.device)
    logging.info("Current config:\n%s", OmegaConf.to_yaml(cfg))


    if cfg.merge_method == "individual":
        logging.info("Running individual checkpoint evaluation...")
        ret = eval_individual_checkpoints(cfg)
    elif cfg.merge_method == "zeroshot":
        logging.info("Running zeroshot checkpoint evaluation...")
        ret = eval_zeroshot(cfg)
    else:
        if cfg.merge_type == "static":
            logging.info("Running static merging evaluation...")
            ret = static_merge_and_eval(cfg)
        elif cfg.merge_type == "adaptive":
            logging.info("Running adaptive merging (TTA) evaluation...")
            ret = adaptive_merge_and_eval(cfg, cfg.device)
        else:
            raise ValueError(f"Unknown merge type: {cfg.merge_type}")

    return ret


def run_parallel_experiments(cfg: DictConfig):
    ray.init(ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
    def run_remote_experiment(path: str):
        try:
            sub_cfg = OmegaConf.load(path)
            merged_cfg = OmegaConf.merge(cfg, sub_cfg)
            return run_experiment(merged_cfg)
        except Exception as e:
            logging.error("Error with config %s: %s", path, str(e))
            return None

    config_dir = cfg.config_list_path
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(os.getcwd(), config_dir)
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    if not yaml_files:
        logging.warning("No YAML files found in %s", config_dir)
        return
    ray_tasks = [run_remote_experiment.remote(yf) for yf in yaml_files]
    results = ray.get(ray_tasks)
    logging.info("All parallel experiments done: %s", results)
