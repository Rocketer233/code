import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import GLUEDataModule
from training import Classifier
import models.opt.convert_checkpoint_opt as convert_checkpoint_opt
import models.opt.modeling_opt as modeling_opt
import models.opt.modeling_opt_gqa as modeling_opt_gqa
from random_search import *
import os
import csv

# tensorboard --logdir ../lightning_logs/ --port 6006

# print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = {'OPT': 'facebook/opt-125m', 'LLaMA': 'JackFram/llama-160m'}
tasks = [['sst2', 2], ['qnli', 2], ['mnli', 3]]

model_name = 'facebook/opt-125m'

res = list()

with open("results/rdsearch_groupings.csv", "w", newline='') as file:
    writer = csv.writer(file)
    for task_name, num_labels in tasks:

        pretrained_model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="/local/scratch/yc538/.cache/huggingface/hub")
        
        model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="/local/scratch/yc538/.cache/huggingface/hub")
        config = model.config
        data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
        data_module.setup("fit")

        classifier = Classifier(model)

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(classifier, data_module)

        groupings, accuracies = randomized_search(model, model_name, task_name, num_labels)
        writer.writerows(groupings)
        writer.writerows([accuracies])
        
        for grouping in groupings:
            model_type = f'OPT-GQA-ACTV1_RDSearch_N=10_#Group={len(grouping["k"][0])}-Pooling'

            config.groups_idx = grouping
            
            state = model.state_dict()
            gqa_model = modeling_opt_gqa.OPTForSequenceClassification(config)
            gqa_model.load_state_dict(convert_checkpoint_opt.mha2gqa(state, grouping, num_heads=12, transpose_layer=True))

            classifier = Classifier(gqa_model)

            logger = TensorBoardLogger("/local/scratch/yc538/lightning_logs", name=task_name + "-" + model_type)
            trainer = pl.Trainer(max_epochs=3, logger=logger)
            trainer.fit(classifier, data_module)

            # total_flops = total_params = 0
            # for _, module in gqa_model.named_modules():
            #     if hasattr(module, "flops"):
            #         total_flops += module.flops
            #     if hasattr(module, "num_params"):
            #         total_params += module.num_params
            # # print(f"total flops: {total_flops}")
            
            # res.append((total_params, total_flops))

        # print(res)

        # with open("results_kvsvdk=128.csv", "w", newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(res)
