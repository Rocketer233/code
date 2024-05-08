import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import GLUEDataModule
from training import OPTClassifier
from opt.configuration_opt_lora import OPTLoraConfig
import opt.convert_checkpoint_opt as convert_checkpoint_opt
import opt.modeling_opt_lora as modeling_opt_lora
import opt.modeling_opt_gqa_lora as modeling_opt_gqa_lora
from lora_utils import (
    print_trainable_parameters,
    mark_only_lora_as_trainable
)
from grouping_metrics import neighbour_grouping
import toml, csv

# tensorboard --logdir lightning_logs/ --port 6006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tasks = [['sst2', 2], ["qnli", 2], ["mnli", 3]]

model_name = 'facebook/opt-125m'
config_files = ["lora_by_type.toml"]
res = []

for config_file in config_files:
    # load toml config file
    with open(config_file, "r") as f:
        lora_config = toml.load(f)
    print(f"LoRA PEFT with {config_file} config file successfully loaded!")

peft_config = OPTLoraConfig.from_pretrained(model_name, lora_config=lora_config, num_labels=3)
print(peft_config.lora_config["default"]["r"])

# lora_config is a dict representation of the toml file

with open("results/opt_lora_params_flops.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for task_name, num_labels in tasks:
        groupings = neighbour_grouping()
        data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
        data_module.setup("fit")
        
        for grouping in groupings:
            model_type = f'OPT-GQA-LoRA_R=32_Neighbouring_#Group={len(grouping["k"][0])}-Pooling'
            
            peft_config = OPTLoraConfig.from_pretrained(model_name, lora_config=lora_config, num_labels=num_labels)
            model = modeling_opt_lora.OPTForSequenceClassification.\
                    from_pretrained(model_name, config=peft_config)

            model.config.groups_idx = grouping
            gqa_model = modeling_opt_gqa_lora.OPTForSequenceClassification(model.config)
            state = model.state_dict()
            gqa_model.load_state_dict(convert_checkpoint_opt.mha2gqa_lora(state, grouping, num_heads=12, transpose_layer=True))
            gqa_model = mark_only_lora_as_trainable(gqa_model)
            print_trainable_parameters(gqa_model)

            classifier = OPTClassifier(gqa_model)

            logger = TensorBoardLogger("/local/scratch/yc538/lightning_logs", name=task_name + "-" + model_type)
            trainer = pl.Trainer(max_epochs=20, logger=logger)
            trainer.fit(classifier, data_module)
            
            total_flops = total_params = 0
            for _, module in gqa_model.named_modules():
                if hasattr(module, "flops"):
                    total_flops += module.flops
                if hasattr(module, "num_params"):
                    total_params += module.num_params
            # print(f"total flops: {total_flops}")
            
            res.append((total_params, total_flops))
    writer.writerows(res)