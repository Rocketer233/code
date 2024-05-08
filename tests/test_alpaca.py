import torch
import transformers
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datamodule import GLUEDataModule,AlpacaDataModule
from training import Classifier
import models.opt.convert_checkpoint_opt as convert_checkpoint_opt
import models.opt.modeling_opt as modeling_opt
import models.opt.modeling_opt_gqa as modeling_opt_gqa
import models.llama.modeling_llama as modeling_llama
import models.llama.modeling_llama_gqa as modeling_llama_gqa
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import copy, time

model_name = 'JackFram/llama-160m'
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  # paper uses 3
LEARNING_RATE = 2e-5  # from the original paper
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data

model = modeling_llama.LlamaForCausalLM.from_pretrained(model_name)
config = model.config
print(config)
data_module = AlpacaDataModule(model_name=model_name)
data_module.setup("fit")

trainer = transformers.Trainer(
    model=model,
    train_dataset=data_module.dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        output_dir="alpaca",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(data_module.tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)