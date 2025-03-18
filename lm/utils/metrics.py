import os
import numpy as np
from transformers import EvalPrediction
import evaluate

from utils.glue_data_loader import glue_data_metrics_map, glue_data_num_labels_map, rev_glue_data_id_map
from utils.load_config import cache_dir


def compute_metrics(eval_pred: EvalPrediction, dataset_names: list):
    if len(dataset_names) > 1:
        results = []
        dataset_ids = eval_pred.label_ids[0]
        for dataset_id in np.unique(dataset_ids):
            single_dataset_indices = dataset_ids == dataset_id
            single_dataset_num_labels = glue_data_num_labels_map[rev_glue_data_id_map[dataset_id.item()]]
            if single_dataset_num_labels > 1:
                predictions = np.argmax(eval_pred.predictions[single_dataset_indices][:, :single_dataset_num_labels], axis=1)
                labels = eval_pred.label_ids[1][single_dataset_indices].astype(np.longlong)
            else:
                assert single_dataset_num_labels == 1, "wrong number of labels!"
                predictions = eval_pred.predictions[single_dataset_indices][:, 0]
                labels = eval_pred.label_ids[1][single_dataset_indices]
            try:
                metric_func = evaluate.load(path=os.path.join(cache_dir, "evaluate/metrics/glue"), config_name=rev_glue_data_id_map[dataset_id.item()])
            except:
                metric_func = evaluate.load(path="glue", config_name=rev_glue_data_id_map[dataset_id.item()], cache_dir=cache_dir)
            result = metric_func.compute(predictions=predictions, references=labels)
            if len(result.keys()) > 1:
                result["averaged_scores"] = np.mean(list(result.values())).item()
            result["dataset_name"] = rev_glue_data_id_map[dataset_id.item()]
            results.append(result)
        dataset_scores = []
        for result in results:
            metric_name = glue_data_metrics_map[result["dataset_name"]]
            dataset_scores.append(result[metric_name])
        return {"averaged_scores": np.mean(dataset_scores).item(), "all_results": results}
    else:
        dataset_name = dataset_names[0]
        if eval_pred.predictions.shape[1] > 1:
            predictions = np.argmax(eval_pred.predictions, axis=1)
        else:
            assert glue_data_num_labels_map[dataset_name] == 1, "wrong number of labels!"
            predictions = np.squeeze(eval_pred.predictions, axis=1)
        try:
            metric_func = evaluate.load(path=os.path.join(cache_dir, "evaluate/metrics/glue"), config_name=dataset_name)
        except:
            metric_func = evaluate.load(path="glue", config_name=dataset_name)
        result = metric_func.compute(predictions=predictions, references=eval_pred.label_ids)
        if len(result.keys()) > 1:
            result["averaged_scores"] = np.mean(list(result.values())).item()
        return result
