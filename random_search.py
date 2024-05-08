import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datamodule import GLUEDataModule
from training import OPTClassifier
import convert_checkpoint
# import opt.modeling_opt as modeling_opt
# import opt.modeling_opt_gqa as modeling_opt_gqa
import llama.modeling_llama as modeling_llama
import llama.modeling_llama_gqa as modeling_llama_gqa
import matplotlib.pyplot as plt
import numpy as np
import copy, time, csv, math
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/opt-125m"

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

def weight_distance(model, task_name, num_labels):
    state = model.state_dict()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    distance = []
    for layer_id in range(num_layers):
        vectors = {}
        vectors["k"] = [[] for _ in range(num_heads)]
        vectors["v"] = [[] for _ in range(num_heads)]
        
        for t in ("k", "v"):
            mat_name = f'model.layers.{layer_id}.self_attn.{t}_proj.weight'
            layer = state[mat_name].transpose(0, 1)
            for i, x in enumerate(torch.tensor_split(layer, num_heads, dim=1)):
                vectors[t][i] = x.transpose(0, 1).flatten()
        
        # for i in range(num_heads):
        #     print(vectors["k"][i].shape, vectors["v"][i].shape)
        
        distance.append({})
        for t in ("k", "v"):
            similarity = cosine_similarity(vectors[t], vectors[t])
            distance[layer_id][t] = 1 - similarity
    
    return distance

def actv_distance1(model, task_name, num_labels, num_samples=300):
    model_name = model.config._name_or_path
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    dataset = load_dataset('glue', task_name,
        cache_dir="/local/scratch/yc538/.cache/huggingface/datasets")
    train_dataset = dataset['train']

    size = num_samples // num_labels
    sampled_idcs = []
    for i in range(num_labels):
        indices = [j for j, example in enumerate(train_dataset) if example['label'] == i]
        sampled_idcs.append(np.random.choice(indices, size=size, replace=False))
    sampled_idcs = np.concatenate(sampled_idcs)
    sampled_dataset = train_dataset.select(sampled_idcs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampled_dataset.set_format(type='torch')
    samples = sampled_dataset.map(
                partial(convert_to_features, tokenizer=tokenizer, task_name=task_name),
                batched=True,
                remove_columns=["label"]
            )
    loader = DataLoader(samples, batch_size=num_samples, shuffle=False)

    print(samples["labels"])
    activation = [{} for _ in range(num_layers)]

    def module_hook(layer_id, t):
        def hook(module, input, output):
            activation[layer_id][t] = output.mean(dim=0)
            # print(input[0].shape, type(output), output.shape)
        return hook

    model = model.to(device)
    model.eval()
    modules = dict([*model.named_modules()])
    for i in range(num_layers):
        for t in ('k', 'v'):
            layer = f'model.layers.{i}.self_attn.{t}_proj'
            modules[layer].register_forward_hook(module_hook(i, t))

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
    
    distance = []
    
    for layer_id in range(num_layers):
        distance.append({})
        for t in ('k', 'v'):
            mat = activation[layer_id][t]
            vectors = []
            for x in torch.tensor_split(mat, num_heads, dim=1):
                y = x.cpu()
                vectors.append(y.flatten() if t == 'k' else y.transpose(0, 1).flatten())
            
            similarity = cosine_similarity(vectors)
            distance[layer_id][t] = 1 - similarity
    
    return distance

def actv_distance2(model, task_name, num_labels, num_samples=300):
    model_name = model.config._name_or_path
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    dataset = load_dataset('glue', task_name,
        cache_dir="/local/scratch/yc538/.cache/huggingface/datasets")
    train_dataset = dataset['train']

    size = num_samples // num_labels
    sampled_idcs = []
    for i in range(num_labels):
        indices = [j for j, example in enumerate(train_dataset) if example['label'] == i]
        sampled_idcs.append(np.random.choice(indices, size=size, replace=False))
    sampled_idcs = np.concatenate(sampled_idcs)
    sampled_dataset = train_dataset.select(sampled_idcs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampled_dataset.set_format(type='torch')
    samples = sampled_dataset.map(
                partial(convert_to_features, tokenizer=tokenizer, task_name=task_name),
                batched=True,
                remove_columns=["label"]
            )
    loader = DataLoader(samples, batch_size=64, shuffle=False)

    print(samples["labels"])
    activation = [{} for _ in range(num_layers)]

    def module_hook(layer_id, t):
        def hook(module, input, output):
            if t not in activation[layer_id]:
                activation[layer_id][t] = output.sum(dim=0)
            else:
                activation[layer_id][t] += output.sum(dim=0)
        return hook

    model = model.to(device)
    model.eval()
    modules = dict([*model.named_modules()])
    for i in range(num_layers):
        for t in ('k', 'v'):
            layer = f'model.layers.{i}.self_attn.{t}_proj'
            modules[layer].register_forward_hook(module_hook(i, t))

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
    
    distance = []
    
    for layer_id in range(num_layers):
        distance.append({})
        for t in ('k', 'v'):
            mat = activation[layer_id][t] / num_samples
            vectors = []
            for x in torch.tensor_split(mat, num_heads, dim=1):
                y = x.cpu()
                vectors.append(y if t == 'k' else y.transpose(0, 1))
            
            distance[layer_id][t] = [[0] * num_heads for _ in range(num_heads)]
            for i in range(num_heads):
                for j in range(num_heads):
                    if i != j:
                        for row1 in vectors[i]:
                            mx = -math.inf
                            for row2 in vectors[j]:
                                mx = max(mx, cosine_similarity(row1.reshape(1, -1), row2.reshape(1, -1))[0, 0])
                            distance[layer_id][t][i][j] += 1 - mx
                            
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    distance[layer_id][t][i][j] = \
                    distance[layer_id][t][j][i] = \
                    (distance[layer_id][t][i][j] + distance[layer_id][t][j][i]) / 2
    
    print(distance[10]['k'])
    return distance

def mutate_grouping(distance, grouping, k=3):
    grouping_list = grouping.flatten()
    
    elem_to_move = np.random.choice(grouping_list)
    
    distances = distance[elem_to_move]
    
    # print(elem_to_move)
    # print(distances)
    
    same_group_elements = grouping[np.where(grouping == elem_to_move)[0][0]]
    for elem in same_group_elements:
        distances[elem] = np.inf  # Set distance to infinity to exclude them
    
    # Find the closest element not in the same group
    closest_elements = np.argsort(distances)[:k]
    closest_element = np.random.choice(closest_elements)
    # print(closest_element)
    # print(closest_elements)
    
    target_idx = np.where(grouping_list == closest_element)[0][0]
    source_idx = np.where(grouping_list == elem_to_move)[0][0]
    
    # print(target_idx, source_idx)
    
    grouping_list[source_idx] = closest_element
    grouping_list[target_idx] = elem_to_move
    
    # print(grouping_list.reshape(grouping.shape))
    return grouping_list.reshape(grouping.shape)

def mutate_grouping2(distance, grouping, k=3):
    grouping_list = grouping.flatten()
    
    elem_to_move = np.random.choice(grouping_list)
    
    distances = distance[elem_to_move]
    
    # print(elem_to_move)
    # print(distances)
    
    same_group_elements = grouping[np.where(grouping == elem_to_move)[0][0]]
    for elem in same_group_elements:
        distances[elem] = np.inf  # Set distance to infinity to exclude them
    
    # Find the closest element not in the same group
    closest_elements = np.argsort(distances)[:k]
    closest_element = np.random.choice(closest_elements)
    # print(closest_element)
    # print(closest_elements)
    
    target_idx = np.where(grouping_list == closest_element)[0][0]
    source_idx = np.where(grouping_list == elem_to_move)[0][0]
    
    # print(target_idx, source_idx)
    
    grouping_list[source_idx] = closest_element
    grouping_list[target_idx] = elem_to_move
    
    # print(grouping_list.reshape(grouping.shape))
    return grouping_list.reshape(grouping.shape)

def get_accuracy(model, gqa_model, layer, grouping, groups, t, loader, num_samples):
    tmp_grouping = copy.deepcopy(grouping)
    tmp_grouping[t][layer] = groups
    
    state = model.state_dict()
    gqa_model.load_state_dict(convert_checkpoint.mha2gqa(state, tmp_grouping, num_heads=12, transpose_layer=True))
    
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


def find_grouping(distance_matrix, model, gqa_model, size, layer, grouping,
                  t, loader, num_samples,
                  num_iterations=10, reset_probability=0.1, acceptance_probability=0.1):
    num_heads = model.config.num_attention_heads
    
    current_groups = np.random.choice(np.arange(0, num_heads), 
                    size=(num_heads // size, size), replace=False)
    
    best_accuracy = get_accuracy(model, gqa_model, layer, grouping, current_groups, t, loader, num_samples)
    best_groups = current_groups
    
    for _ in range(num_iterations):
        # With a certain probability, reset to a completely random grouping
        if np.random.rand() < reset_probability:
            current_groups = np.random.choice(np.arange(0, num_heads), 
                    size=(num_heads // size, size), replace=False)

        new_groups = mutate_grouping(distance_matrix, np.copy(current_groups))
        
        accuracy = get_accuracy(model, gqa_model, layer, grouping, new_groups, t, loader, num_samples)
        
        print(accuracy)
        # Accept new grouping based on accuracy improvement or acceptance probability
        if accuracy > best_accuracy or np.random.rand() < acceptance_probability:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_groups = new_groups
            current_groups = new_groups  # Move to new grouping even if not the best
    
    return best_groups, best_accuracy

def find_asym_grouping(distance_matrix, model, gqa_model, size, layer, grouping,
                  t, loader, num_samples,
                  num_iterations=10, reset_probability=0.1, acceptance_probability=0.1, preserve_probability=0.2):
    num_heads = model.config.num_attention_heads
    
    current_groups = np.random.choice(np.arange(0, num_heads), 
                    size=(num_heads // size, size), replace=False)
    
    best_accuracy = get_accuracy(model, gqa_model, layer, grouping, current_groups, t, loader, num_samples)
    best_groups = current_groups
    
    for _ in range(num_iterations):
        # With a certain probability, reset to a completely random grouping
        if np.random.rand() < reset_probability:
            current_groups = np.random.choice(np.arange(0, num_heads), 
                    size=(num_heads // size, size), replace=False)

        new_groups = mutate_grouping2(distance_matrix, np.copy(current_groups))
        
        accuracy = get_accuracy(model, gqa_model, layer, grouping, new_groups, t, loader, num_samples)
        
        print(accuracy)
        # Accept new grouping based on accuracy improvement or acceptance probability
        if accuracy > best_accuracy or np.random.rand() < acceptance_probability:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_groups = new_groups
            current_groups = new_groups  # Move to new grouping even if not the best
    
    return best_groups, best_accuracy

def randomized_search(model, model_name, task_name, num_labels):
    # model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
    # data_module.setup("fit")

    # classifier = OPTClassifier(model)

    # trainer = pl.Trainer(max_epochs=1)
    # trainer.fit(classifier, data_module)
    
    dataset = load_dataset('glue', task_name,
        cache_dir="/local/scratch/yc538/.cache/huggingface/datasets")
    split = ('validation' if task_name != 'mnli' else 'validation_matched')
    validation = dataset[split]

    if task_name != 'mnli':
        sampled_dataset = validation
    else:
        num_samples = 3000
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
    
    distance = weight_distance(model, task_name, num_labels)
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
     
    nums_groups = [4, 3, 6, 2]
    groupings = []
    
    config = model.config
    accuracies = []
    for i in range(len(nums_groups)):
        accuracy = 0
        grouping = [[[i] for i in range(num_heads)]] * num_layers
        grouping = {'k': grouping, 'v': grouping}
        size = num_heads // nums_groups[i]
        for j in range(num_layers):
            for t in ['k', 'v']:
                grouping[t][j] = [range(i, i + size) for i in range(0, num_heads, size)]
            
                config.groups_idx = grouping
                gqa_model = modeling_llama_gqa.LlamaForSequenceClassification(config)
                
                x, accuracy = find_grouping(distance[j][t], model, gqa_model, size, j, grouping, t, loader, len(validation))
                grouping[t][j] = x
                print(task_name, i, j, t, "accuracy", accuracy)
        groupings.append(grouping)
        accuracies.append(accuracy)
    
    return groupings, accuracies

if __name__ == "__main__":
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    # model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # actv_distance2(model, task_name, num_labels)