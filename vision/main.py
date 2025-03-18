# src/experiments/main.py
import os
import glob
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils
from runner import run_experiment, run_parallel_experiments
from src.utils import gpu_supports_bf16_via_pytorch

@hydra.main(config_path="configs", config_name="hydra_default", version_base=None)
def main(cfg: DictConfig):
    logging.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    if cfg.get("half_precision", False):
       assert gpu_supports_bf16_via_pytorch(), "GPU does not support BF16"
    
    if cfg.get("do_parallel", False):
        run_parallel_experiments(cfg)
    else:
        config_list_path = cfg.config_list_path
        if not os.path.isabs(config_list_path):
            config_list_path = os.path.join(hydra.utils.get_original_cwd(), config_list_path)
        if os.path.isdir(config_list_path):
            yaml_files = glob.glob(os.path.join(config_list_path, "*.yaml"))
            if not yaml_files:
                logging.warning("No YAML config files found in directory: %s", config_list_path)
                run_experiment(cfg)
            else:
                for yf in yaml_files:
                    logging.info("Running experiment for config file: %s", yf)
                    sub_cfg = OmegaConf.load(yf)
                    merged_cfg = OmegaConf.merge(cfg, sub_cfg)
                    run_experiment(merged_cfg)
        elif os.path.isfile(config_list_path) and config_list_path.endswith(".yaml"):
            logging.info("Running experiment for single config file: %s", config_list_path)
            sub_cfg = OmegaConf.load(config_list_path)
            merged_cfg = OmegaConf.merge(cfg, sub_cfg)
            run_experiment(merged_cfg)
        else:
            logging.warning("config_list_path %s is neither a directory nor a YAML file. Running base config.", config_list_path)
            run_experiment(cfg)

if __name__ == "__main__":
    main()
