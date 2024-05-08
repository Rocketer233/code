import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import GLUEDataModule
from training import OPTClassifier
from opt.configuration_opt_lora import OPTLoraConfig
import opt.modeling_opt as modeling_opt
import opt.modeling_opt_lora as modeling_opt_lora
from lora_utils import (
    print_trainable_parameters,
    mark_only_lora_as_trainable
)
import toml

# tensorboard --logdir lightning_logs/ --port 6006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'OPT-LoRA'
model_name = 'facebook/opt-125m'
task_name = 'mnli'
num_labels = 3
config_files = ["lora_by_type.toml"]

for config_file in config_files:
    # load toml config file
    with open(config_file, "r") as f:
        lora_config = toml.load(f)
    print(f"LoRA PEFT with {config_file} config file successfully loaded!")

# lora_config is a dict representation of the toml file
# print(lora_config)

peft_config = OPTLoraConfig.from_pretrained(model_name, lora_config=lora_config, num_labels=num_labels)
# print(peft_config)
model = modeling_opt_lora.OPTForSequenceClassification.\
        from_pretrained(model_name, config=peft_config)
model = mark_only_lora_as_trainable(model)
print_trainable_parameters(model)


data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
data_module.setup("fit")

classifier = OPTClassifier(model)

logger = TensorBoardLogger("lightning_logs", name=task_name + "-" + model_type)
trainer = pl.Trainer(max_epochs=30, logger=logger)
trainer.fit(classifier, data_module)