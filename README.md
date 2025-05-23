# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging
This is the official repository for paper **[AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](https://arxiv.org/abs/2503.22178).** 

![Image](https://github.com/user-attachments/assets/96fff66d-99d0-408a-a371-7fe309fab7a2)

#### Updates
**2025-04-30**: We share checkpoint links which depedencies are removed for vision tasks. Memory saving options for experiments are added.

## Abstract
_Model merging has emerged as a promising approach for unifying independently fine-tuned models into an integrated framework, significantly enhancing computational efficiency in multi-task learning. Recently, several SVD-based techniques have been introduced to exploit low-rank structures for enhanced merging, but their reliance on such manually designed rank selection often leads to cross-task interference and suboptimal performance. In this paper, we propose **AdaRank**, a novel model merging framework that adaptively selects the most beneficial singular directions of task vectors to merge multiple models. We empirically show that the dominant singular components of task vectors can cause critical interference with other tasks, and that naive truncation across tasks and layers degrades performance. In contrast, AdaRank dynamically prunes the singular components that cause interference and offers an optimal amount of information to each task vector by learning to prune ranks during test-time via entropy minimization. Our analysis demonstrates that such method mitigates detrimental overlaps among tasks, while empirical results show that AdaRank consistently achieves state-of-the-art performance with various backbones and number of tasks, reducing the performance gap between fine-tuned models to nearly 1\%._



## Preparation
### Install Dependencies
Install the required dependencies(Python 3.8+ recommended):
```bash
pip install -r requirements.txt
```
**Note**: This codebase is based on PyTorch 2.2.1 and CUDA 12.1. Ensure your environment meets these requirements for compatibility. Some datasets require a Kaggle API key for loading. Please add your Kaggle API key to the `~/.kaggle/` folder. Refer to the [Kaggle API documentation](https://www.kaggle.com/docs/api) for instructions on obtaining and setting up your API key.

---
### Datasets
- **Vision Datasets**: 
Most of the datasets will be downloaded automatically once you first execute the code. For issues, follow [TALL-Masks](https://github.com/nik-dim/tall_masks).

- **Language Model Datasets**:
Refer to [DARE](https://github.com/yule-BUAA/MergeLM) & [EMR-Merging](https://github.com/harveyhuang18/EMR_Merging/).

  - **Note**: Update `cache_dir` in `utils/load_config.py` to point to your dataset directory.

### Checkpoints
**Vision Experiments**:
  - 8 Tasks: [ViT-B-32/ViT-B-32_TA](https://huggingface.co/JH-C-k/ViT-B-32/tree/main) / [ViT-L-14_TA](https://huggingface.co/JH-C-k/ViT-L-14_TA/tree/main)
  - 14/20 Tasks:  [ViT-B-32/ViT-B-32](https://huggingface.co/JH-C-k/ViT-B-32/tree/main) / [ViT-L-14](https://huggingface.co/JH-C-k/ViT-L-14/tree/main)
  - Place your checkpoints as below:
```bash
your_directory
├── ViT-B-32_TA # 8 task checkpoints 
│   ├── Cars
│   ├── DTD
│   ├── ...
├── ViT-B-32 # 20 task checkpoints 
│    ├── Cars
│    ├── DTD
│    ├── ...
│    ├── FER2013
│    ├── ...
├── ViT-L-14_TA
└── ViT-L-14
```
  - Checkpoints are modified from finetuned checkpoints from [Task Arithmetic](https://github.com/mlfoundations/task_vectors) (8 Tasks) and [TALL-Masks](https://github.com/nik-dim/tall_masks) (20 Tasks) with removing the directory dependence.
  - Since checkpoints for 8 tasks and 20 tasks are differently finetuned, we provide options for benchmarks. If you set `TA_MODE=True` in the exp config, it will automatically fetch 8 tasks checkpoints from `ViT-{B or L}-{32 or 14}_TA` folder and use them for 8 tasks evaluation. If you set `False`, all checkpoints from TALL-Masks are used.

**Language Models**:
  - RoBERTa: [vanillaOVO/roberta_base_glue_ckpts](https://huggingface.co/vanillaOVO/roberta_base_glue_ckpts/tree/main).
  - GPT-2: [tanganke/gpt-2-models](https://huggingface.co/collections/tanganke/gpt-2-models-fine-tuned-on-tasks-from-glue-benchmark-664ab37d9e33e622679f541b).
    -  **Note**: Update `weight_dir` in `utils/load_config.py` to your directory, and download the checkpoints under this directory. The folder should look like below:
```bash
your_directory (weight_dir)
├── roberta
│   ├── cola
│   ├── mnli
│   ├── ...
└── gpt2
    ├── gpt2_cola
    ├── gpt2_mnli
    ├── ...
``` 
## Running Experiments
### Vision Model Merging

You could find the default config in `/vision/configs/hydra_default.yaml`. Once you prepare the checkpoints and datasets, modify these options:

```python
# Data Path Configuration
data_location: "/path/to/your/dataset/folders/"  
weight_root: "/path/to/your/checkpoints/"  
save: "/path/to/your/checkpoints/" # Used for loading task heads; recommended to set in the same folder as checkpoints.  
openclip_cachedir: "/path/to/your/checkpoints" # Detects pretrained OpenCLIP checkpoints; downloads compatible versions if absent.
```
Child configs for each experiment are saved in their respective folders.

Note: If options overlap between default and experiment configs, the experiment config values take precedence. We recommend maintaining the default config and editing child configs for experiment control.
Run experiments for merging 8, 14, or 20 tasks with ViT-B/32 or ViT-L/14 backbones:

- **8-Task Benchmark**:
  ```bash
  python ./vision/main.py config_list_path="./configs/adarank_{B or L}_T8/adarank_{TA or CART}.yaml"
  ```
- **14-Task Benchmark**:
  ```bash
  python ./vision/main.py config_list_path="./configs/adarank_{B or L}_T14/adarank_{TA or CART}.yaml"
  ```
- **20-Task Benchmark**:
  ```bash
  python ./vision/main.py config_list_path="./configs/adarank_{B or L}_T20/adarank_{TA or CART}.yaml"
  ```
We provide options for two baseline methods, Task Arithmetic(TA) and [CART](https://arxiv.org/abs/2412.12153).
### Language Model Merging

Run experiments for merging 7 tasks:

- **RoBERTa**:
  ```bash
  python ./lm/adarank_roberta_glue.py --exp_config="./config/roberta_{TA or CART}.yaml"
  ```
- **GPT-2**:
  ```bash
  python ./lm/adarank_gpt_glue.py --exp_config="./config/gpt_{TA or CART}.yaml"
  ```

## Citation

If you use this code in your research, we would be grateful to cite our paper:

```bibtex
[@misc{lee2025adarankadaptiverankpruning,
      title={AdaRank: Adaptive Rank Pruning for Enhanced Model Merging}, 
      author={Chanhyuk Lee and Jiho Choi and Chanryeol Lee and Donggyun Kim and Seunghoon Hong},
      year={2025},
      eprint={2503.22178},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.22178}, 
}]
```

## Acknowledgement
This repository is built upon codebase of [Task Arithmetic](https://github.com/mlfoundations/task_vectors), [AdaMerging](https://github.com/EnnengYang/AdaMerging), and [EMR-Merging](https://github.com/harveyhuang18/EMR_Merging) (especially for language model experiments). Thanks to the authors for sharing their work.