##  Codebase for ICCV sumission "AdaRank: Adaptive Rank Pruning for Enhanced Model Merging"
### Preparing Datasets and Checkpoints
#### Datasets: 
Vision Datasets
-  Refer to Wang et al. https://github.com/nik-dim/tall_masks

Language Model Datasets
- Refer to Yu et al. https://github.com/yule-BUAA/MergeLM
- Refer to Huang et al. https://github.com/harveyhuang18/EMR_Merging/
- You should replace ```cache_dir``` in ```utils/load_config.py``` to your directory path that contains dataset.

#### Checkpoints: 
Vision Experiment
- 8 Tasks: Refer to Ilharco et al. https://github.com/mlfoundations/task_vectors
- 14/20 Tasks: Refer to Wang et al. https://github.com/nik-dim/tall_masks

Language Model
- Roberta: https://huggingface.co/vanillaOVO/roberta_base_glue_ckpts/tree/main
- GPT2: https://huggingface.co/collections/tanganke/gpt-2-models-fine-tuned-on-tasks-from-glue-benchmark-664ab37d9e33e622679f541b
- You should replace ```weight_dir``` in ```utils/load_config.py``` to your directory path that contains weight.

### Vision Model Merging Experiment

For vision model merging experiments, we provide codes for merging 8, 14, 20 tasks for 2 backbones (ViT-B/32 and ViT-L/14), respectively.

8-Tasks Benchmark: ```python ./vision/main.py config_list_path="./configs/adarank_{B or L}_T8/adarank_{TA or CART}.yaml"```

14-Tasks Benchmark: ```python ./vision/main.py config_list_path="./configs/adarank_{B or L}_T14/adarank_{TA or CART}.yaml"```

20-Tasks Benchmark: ```python ./vision/main.py config_list_path="./configs/adarank_{B or L}_T20/adarank_{TA or CART}.yaml"```


### Language Model Merging Experiment

For language model experiments, we provide codes for merging 7 tasks for ViT-B/32.

RoBERTa: ```python ./lm/adarank_roberta_glue.py --exp_config="./config/roberta_{TA or CART}.yaml"```

GPT2: ```python ./lm/adarank_gpt_glue.py --exp_config="./config/gpt_{TA or CART}.yaml"```


