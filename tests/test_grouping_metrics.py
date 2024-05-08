import torch
import models.opt.modeling_opt as modeling_opt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from grouping_metrics import *

def test1():
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    kv_svd_grouping(model)
    
def test2():
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    neighbour_grouping()

def test3():
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    kv_grouping(model)

def test4():
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    actv_grouping(model)