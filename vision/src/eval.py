import os
import json
import tqdm

import torch
import numpy as np

from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.heads import get_classification_head
from src.models.modeling import ImageClassifier

from src.datasets.registry import get_dataset


def eval_single_dataset_with_prediction(image_encoder, dataset_name, dataloader, args, classification_head=None,):
    if classification_head == None:
        classification_head = get_classification_head(args, dataset_name)

    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        loss = 0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            
            if args.half_precision:
                x = data['images'].to(torch.bfloat16).to(device)
            else:
                x = data['images'].to(device)
                
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model, dataset_name, args)
            
            # loss = torch.nn.functional.cross_entropy(logits, y)
            pred = logits.argmax(dim=1, keepdim=True).to(device)

            if i == 0:
                all_preds = pred
                all_labels = y
            else:
                all_preds = torch.cat((all_preds, pred), dim=0)
                all_labels = torch.cat((all_labels, y), dim=0)
            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')

    return metrics, all_preds, all_labels

def eval_single_dataset_with_loss(image_encoder, classification_head, dataset_name, args, dataloader, num_batches=1):
    if classification_head is None:
        classification_head = get_classification_head(args, dataset_name)
    
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    device = args.device
    print(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # if i >= num_batches:
        #     break
        data = maybe_dictionarize(data)
        
        x = data['images'].to(device)
        y = data['labels'].to(device)

        logits = utils.get_logits(x, model, dataset_name, args)
        
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        total_correct += pred.eq(y.view_as(pred)).sum().item()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    top1 = total_correct / total_samples if total_samples > 0 else 0
    
    metrics = {'top1': top1, 'loss': avg_loss}
    print(f"Done evaluating on {dataset_name}. Acc: {top1*100}, Loss: {avg_loss:.4f}")
    return metrics



def eval_single_dataset(image_encoder, dataset_name, args, classification_head=None):
    if classification_head is None:
        classification_head = get_classification_head(args, dataset_name)
    
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.eval_batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
        
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model, dataset_name, args)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')

    return metrics

def eval_adamerge_single_dataset(args, adamerge_model, dataset_name):
    dataset = get_dataset(
        dataset_name,
        adamerge_model.pretrained_model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset,
        is_train=False,
        args=args,
    )
    device = args.device
    adamerge_model.pretrained_model.eval()
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            
            if args.half_precision:
                x = data['images'].to(torch.bfloat16).to(device)
            else:
                x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = adamerge_model(inp = x, dataset_name=dataset_name)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info
