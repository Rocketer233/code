import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datamodule import GLUEDataModule
from training import OPTClassifier
import models.opt.convert_checkpoint_opt as convert_checkpoint_opt
import models.opt.modeling_opt as modeling_opt
import models.opt.modeling_opt_gqa as modeling_opt_gqa
import models.llama.modeling_llama as modeling_llama
import models.llama.modeling_llama_gqa as modeling_llama_gqa
from eval_prompting import evaluate
import matplotlib.pyplot as plt
import numpy as np
import copy, time, csv
from functools import partial



def convert_to_features(examples, tokenizer, task_name):
    if task_name == "sst2":
        texts = examples['sentence']
    elif task_name == "mnli":
        texts = list(zip(examples['premise'], examples['hypothesis']))
    else: texts = list(zip(examples['question'], examples['sentence']))

    features = tokenizer(texts, padding='max_length', 
                truncation=True, max_length=128)

    features["labels"] = examples["label"]
    
    for k in features:
        features[k] = torch.tensor(features[k])
        
    print(features.keys())

    return features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_layers = 12
num_heads = 12


def get_accuracy1(model, gqa_model, layer, grouping, group, loader, num_samples):
    groups = [group] + [[i] for i in range(num_heads) if i not in group]
    tmp_grouping = copy.deepcopy(grouping)
    tmp_grouping[layer] = groups
    tmp_grouping = {'k': tmp_grouping, 'v': tmp_grouping}
    
    state = model.state_dict()
    gqa_model.load_state_dict(convert_checkpoint_opt.mha2gqa(state, tmp_grouping, num_heads=12, transpose_layer=True))
    
    gqa_model.to(device)
    gqa_model.eval()
    accuracy = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = gqa_model(input_ids, attention_mask=attention_mask)
        
        logits = outputs['logits']
        labels = batch['labels'].to(device)

        _, predicted = torch.max(logits, 1)

        correct_predictions = (predicted == labels).sum().item()

        accuracy += correct_predictions

    accuracy /= num_samples

    return accuracy

def get_accuracy2(model, gqa_model, layer, grouping, group, loader, num_samples):
    groups = [group] + [[i] for i in range(num_heads) if i not in group]
    tmp_grouping = copy.deepcopy(grouping)
    tmp_grouping[layer] = groups
    tmp_grouping = {'k': tmp_grouping, 'v': tmp_grouping}
    
    state = model.state_dict()
    gqa_model.load_state_dict(convert_checkpoint_opt.mha2gqa(state, tmp_grouping, num_heads=12, transpose_layer=True))
    
    gqa_model.to(device)
    gqa_model.eval()
    return evaluate(gqa_model, ["mmlu"])


def dfs(cur, heads, i, size, layer, grouping, model, gqa_model, loader, num_samples):
    if len(cur) == size:
        return (get_accuracy(model, gqa_model, layer, grouping, cur, loader, num_samples), cur)
    if i == len(heads):
        return (0, 0)
    
    (a1, b1) = dfs(cur, heads, i + 1, size, layer, grouping, model, gqa_model, loader, num_samples)
    (a2, b2) = dfs(cur + [heads[i]], heads, i + 1, size, layer, grouping, model, gqa_model, loader, num_samples)
    
    return ((a1, b1) if a1 > a2 else (a2, b2))

    
def find_grouping(model, loader, size, num_samples):
    grouping = [[[i] for i in range(num_heads)]] * num_layers
    print(grouping)
    config = model.config
    
    accuracy = 0
    for i in range(num_layers):
        heads = list(range(num_heads))
        grouping[i] = [range(size)] + [[i] for i in range(size, num_heads)]
        config.groups_idx = {'k': grouping, 'v': grouping}
        gqa_model = modeling_opt_gqa.OPTForSequenceClassification(config)
        groups = []
        while len(heads) > 0:
            accuracy, group = dfs([heads[0]], heads[1: ], 0, size, i, grouping, model, gqa_model, loader, num_samples)
            print("group = ", group)
            groups.append(group)
            heads = [i for i in heads if i not in group]
        
        grouping[i] = groups
        
        print(groups)
    
    return {'k': grouping, 'v': grouping}


def bf_search(model_name, task_name, num_labels):
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
    data_module.setup("fit")

    classifier = OPTClassifier(model)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(classifier, data_module)
    
    dataset = load_dataset('glue', task_name,
        cache_dir="/local/scratch/yc538/.cache/huggingface/datasets")
    split = ('validation' if task_name != 'mnli' else 'validation_matched')
    validation = dataset[split]

    num_samples = 500
    size = num_samples // num_labels
    sampled_idcs = []
    for i in range(num_labels):
        indices = [j for j, example in enumerate(validation) if example['label'] == i]
        sampled_idcs.append(np.random.choice(indices, size=size, replace=False))
    sampled_idcs = np.concatenate(sampled_idcs)
    sampled_dataset = validation.select(sampled_idcs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampled_dataset.set_format(type='torch')
    samples = sampled_dataset.map(
                partial(convert_to_features, tokenizer=tokenizer, task_name=task_name),
                batched=True,
                remove_columns=["label"]
            )
    loader = DataLoader(samples, batch_size=256, shuffle=False)

    # print(samples["labels"])
    
    start_time = time.time()

    nums_groups = [6, 4, 3, 2]
    groupings = [find_grouping(model, loader, 12 // i, len(samples)) for i in nums_groups]

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
    
    return groupings


# if __name__ == "__main__":
#     model_name = "facebook/opt-125m"
#     tasks = [['mnli', 3]]
    
#     with open("bfsearch_groupings.csv", "w", newline='') as file:
#         writer = csv.writer(file)
#         for task_name, num_labels in tasks:
#             writer.writerows(bf_search(model_name, task_name, num_labels))
    
    # grouping = [[[i] for i in range(num_heads)]] * num_layers
    # groupings = [({"k": grouping, "v": grouping}, 0.5)] * 4
    
    # with open("bfsearch_groupings.csv", "w", newline='') as file:
    #     writer = csv.writer(file)
    #     for i in range(3):
    #         t = [[i]]
    #         writer.writerows(groupings)