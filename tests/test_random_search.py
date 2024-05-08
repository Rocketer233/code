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
from random_search import *

def test1():
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    weight_distance(model, task_name, num_labels)

def test2():
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    actv_distance1(model, task_name, num_labels)

def test3():
    model_name = 'facebook/opt-125m'
    task_name = 'sst2'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    actv_distance2(model, task_name, num_labels)

