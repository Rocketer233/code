import numpy as np
import lm_eval
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
from eval_prompting import *

if __name__ == "__main__":
    model_name = 'meta-llama/Llama-2-7b-hf'
    task_name = "sst2"
    num_labels = 3
    num_samples = 300
    model = modeling_llama.LlamaForCausalLM.from_pretrained(model_name, cache_dir="/local/scratch/yc538/.cache/huggingface/hub").cuda()
    evaluate(model, ["mmlu"])
