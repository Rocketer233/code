import torch
import models.opt.modeling_opt as modeling_opt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import math


def uniform_groups(size, tot=12):
    return [list(range(i, min(i + size, tot))) for i in range(0, tot, size)]

def neighbour_grouping():
    nums_groups = [6, 4, 3, 2]
    groupings = [{"k": [], "v": []} for _ in range(len(nums_groups))]
    for i in range(len(nums_groups)):
        groups = [uniform_groups(12 // nums_groups[i]) for _ in range(12)]
        groupings[i]["k"] = groupings[i]["v"] =  groups
    
    for grouping in groupings:
        print(grouping)
    return groupings

def kv_grouping(model):
    state = model.state_dict()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    nums_groups = [6, 4, 3, 2]
    groupings = [{"k": [], "v": []} for _ in range(len(nums_groups) + 1)]
    
    for layer_id in range(num_layers):
        vectors = {}
        vectors["k"] = [[] for _ in range(num_heads)]
        vectors["v"] = [[] for _ in range(num_heads)]
        
        for t in ("k", "v"):
            mat_name = f'model.decoder.layers.{layer_id}.self_attn.{t}_proj.weight'
            layer = state[mat_name].transpose(0, 1)
            for i, x in enumerate(torch.tensor_split(layer, num_heads, dim=1)):
                vectors[t][i] = (x.flatten() if t == 'k' else x.transpose(0, 1).flatten())
            
            # mat_name = f'model.decoder.layers.{layer_id}.self_attn.{t}_proj.bias'
            # layer = state[mat_name]
            # for i, x in enumerate(torch.tensor_split(layer, num_heads)):
            #     vectors[t][i].append(x.flatten())
            
            # for i in range(num_heads):
            #     vectors[t][i] = torch.concatenate([vectors[t][i][0],
            #                                       vectors[t][i][1]])
        
        for i in range(num_heads):
            print(vectors["k"][i].shape, vectors["v"][i].shape)
        
        for t in ("k", "v"):
            similarity = cosine_similarity(vectors[t], vectors[t])
            distance = 1 - similarity
            print(distance)
            Z = linkage(distance, 'ward')
            
            j = 0
            tree = set(range(num_heads))
            nodes = [[i] for i in range(num_heads)]
            groupings[0][t].append(uniform_groups(1))
            for i, z in enumerate(Z):
                x, y = int(z[0]), int(z[1])
                nodes.append(nodes[x] + nodes[y])
                tree.remove(x)
                tree.remove(y)
                tree.add(num_heads + i)
                if j < len(nums_groups) and len(tree) == nums_groups[j]:
                    groups = [nodes[i] for i in tree]
                    groupings[j + 1][t].append(groups)
                    j += 1
    
    for grouping in groupings:
        print(grouping)
    
    return groupings


def clustering(distance, groupings, nums_groups, num_heads, t, flag = True):
    Z = linkage(distance, 'ward')
    j = 0
    tree = set(range(num_heads))
    nodes = [[i] for i in range(num_heads)]
    
    if flag:
        groupings[0][t].append(uniform_groups(1))
    
    for i, z in enumerate(Z):
        x, y = int(z[0]), int(z[1])
        nodes.append(nodes[x] + nodes[y])
        tree.remove(x)
        tree.remove(y)
        tree.add(num_heads + i)
        if j < len(nums_groups) and len(tree) == nums_groups[j]:
            groups = [nodes[i] for i in tree]
            groupings[j + 1][t].append(groups)
            j += 1


def kv_svd_grouping(model):
    state = model.state_dict()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    nums_groups = [6, 4, 3, 2]
    groupings = [{"k": [], "v": []} for _ in range(len(nums_groups) + 1)]
    
    for layer_id in range(num_layers):
        SVD = {}
        SVD["k"] = [[] for _ in range(num_heads)]
        SVD["v"] = [[] for _ in range(num_heads)]
        
        k = [128, 0, 16]
        
        for t in ("k", "v"):
            mat_name = f'model.decoder.layers.{layer_id}.self_attn.{t}_proj.weight'
            layer = state[mat_name].transpose(0, 1)
            for i, x in enumerate(torch.tensor_split(layer, num_heads, dim=1)):
                SVD[t][i] = torch.linalg.svd(x)
                
            distance = [[0] * num_heads for _ in range(num_heads)]
            
            for x in range(num_heads):
                for y in range(num_heads):
                    for idx in (0, 2):
                        for i in range(16):
                            mx = -math.inf
                            for j in range(16):
                                mx = max(mx, cosine_similarity(
                                            SVD[t][x][idx][:, i].reshape(1, -1), 
                                            SVD[t][y][idx][:, j].reshape(1, -1))[0][0])
                            distance[x][y] += 1 - mx
            
            for x in range(num_heads):
                distance[x][x] = 0
                for y in range(x + 1, num_heads):
                    distance[x][y] = distance[y][x] = (distance[x][y] + distance[y][x]) / 2
            
            # print(distance)
            
            Z = linkage(squareform(distance), 'ward')
            
            j = 0
            tree = set(range(num_heads))
            nodes = [[i] for i in range(num_heads)]
            groupings[0][t].append(uniform_groups(1))
            for i, z in enumerate(Z):
                x, y = int(z[0]), int(z[1])
                nodes.append(nodes[x] + nodes[y])
                tree.remove(x)
                tree.remove(y)
                tree.add(num_heads + i)
                if j < len(nums_groups) and len(tree) == nums_groups[j]:
                    groups = [nodes[i] for i in tree]
                    groupings[j + 1][t].append(groups)
                    j += 1
    
    for grouping in groupings:
        print(grouping)
    
    return groupings


def convert_to_features(examples, task_name, tokenizer):
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
        
    # print(features.keys())

    return features


def actv_grouping(model, task_name, num_labels, num_samples=120):
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
    tokenized_samples = convert_to_features(sampled_dataset, task_name, tokenizer)

    print(tokenized_samples["labels"])
    activation = [{} for _ in range(num_layers)]

    def module_hook(layer_id, t):
        def hook(module, input, output):
            activation[layer_id][t] = output
            print(input[0].shape, type(output), output.shape)
        return hook

    modules = dict([*model.named_modules()])
    for i in range(num_layers):
        for t in ('k', 'v'):
            layer = f'model.decoder.layers.{i}.self_attn.{t}_proj'
            modules[layer].register_forward_hook(module_hook(i, t))

    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_samples)
    
    nums_groups = [6, 4, 3, 2]
    groupings = [{"k": [], "v": []} for _ in range(len(nums_groups) + 1)]
    
    for layer_id in range(num_layers):
        for t in ('k', 'v'):
            mat = activation[layer_id][t].mean(dim=0)
            vectors = [x.flatten() for x in torch.tensor_split(mat, num_heads, dim=1)]
            
            similarity = cosine_similarity(vectors)
            distance = 1 - similarity
            
            clustering(distance, groupings, nums_groups, num_heads, t)
    
    for grouping in groupings:
        print(grouping)
    
    return groupings


if __name__ == "__main__":
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    kv_svd_grouping(model)
    