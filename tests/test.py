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
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import copy, time

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

    return features

model_name = 'facebook/opt-1.3b'
task_name = "sst2"
num_labels = 3
num_samples = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="/local/scratch/yc538/.cache/huggingface/hub")

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
data_module.setup("fit")

classifier = OPTClassifier(model)

trainer = pl.Trainer(max_epochs=1)
trainer.fit(classifier, data_module)

dataset = load_dataset('glue', task_name,
    cache_dir="/local/scratch/yc538/.cache/huggingface/datasets")
split = ('validation' if task_name != 'mnli' else 'validation_matched')
validation = dataset[split]

size = num_samples // num_labels
sampled_idcs = []
for i in range(num_labels):
    indices = [j for j, example in enumerate(validation) if example['label'] == i]
    sampled_idcs.append(np.random.choice(indices, size=size, replace=False))
sampled_idcs = np.concatenate(sampled_idcs)
sampled_dataset = validation

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sampled_dataset.set_format(type='torch')
samples = sampled_dataset.map(
            partial(convert_to_features, tokenizer=tokenizer, task_name=task_name),
            batched=True,
            remove_columns=["label"]
        )
print("type", type(samples))
loader = DataLoader(samples, batch_size=1024, shuffle=False)

start_time = time.time()

model.to(device)
model.eval()
accuracy = 0
for batch in loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs['logits']
    labels = batch['labels'].to(device)

    _, predicted = torch.max(logits, 1)

    correct_predictions = (predicted == labels).sum().item()

    accuracy += correct_predictions

accuracy /= len(validation)
print(accuracy)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")

model_names = ["facebook/opt-1.3b"]

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/local/scratch/yc538/.cache/huggingface/hub")
    for num_labels in [2, 3]:
        model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="/local/scratch/yc538/.cache/huggingface/hub")
        model = modeling_llama.LlamaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="/local/scratch/yc538/.cache/huggingface/hub")

for task_name in ["sst2", "qnli", "mnli"]:
    dataset = load_dataset('glue', task_name, cache_dir="/local/scratch/yc538/.cache/huggingface/datasets")

model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="/local/scratch/yc538/.cache/huggingface/hub")
model.save_pretrained("/local/scratch/yc538/models")