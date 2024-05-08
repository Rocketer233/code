import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import GLUEDataModule
from training import OPTClassifier
from llama.configuration_llama_lora import LlamaLoraConfig
import llama.modeling_llama as modeling_llama
import llama.modeling_llama_lora as modeling_llama_lora
from lora_utils import (
    print_trainable_parameters,
    mark_only_lora_as_trainable
)
import toml

# tensorboard --logdir lightning_logs/ --port 6006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'LLaMA-LoRA'
model_name = 'JackFram/llama-160m'
task_name = 'qnli'
num_labels = 2
config_files = ["lora_by_type.toml"]

for config_file in config_files:
    # load toml config file
    with open(config_file, "r") as f:
        lora_config = toml.load(f)
    print(f"LoRA PEFT with {config_file} config file successfully loaded!")

# lora_config is a dict representation of the toml file
# print(lora_config)

peft_config = LlamaLoraConfig.from_pretrained(model_name, lora_config=lora_config, num_labels=num_labels)
# print(peft_config)
model = modeling_llama_lora.LlamaForSequenceClassification.\
        from_pretrained(model_name, config=peft_config)
model = mark_only_lora_as_trainable(model)
# print_trainable_parameters(model)


data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
data_module.setup("fit")

classifier = OPTClassifier(model)

logger = TensorBoardLogger("lightning_logs", name=task_name + "-" + model_type)
trainer = pl.Trainer(max_epochs=30, logger=logger)
trainer.fit(classifier, data_module)