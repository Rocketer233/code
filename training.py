import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy, MeanMetric
import pdb

class OPTForSequenceClassification(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(base_model.config.vocab_size, num_labels)
        print(base_model.config)
        # print(self.classifier)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        # hidden_states: [B * L * H], L: Sequence length, H: Hidden size
        logits = self.classifier(hidden_states[:, -1])
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


class Classifier(pl.LightningModule):
    def __init__(self, model, lr = 2e-5):
        super().__init__()
        self.model = model
        self.num_labels = model.num_labels
        self.lr = lr
        self.acc_train = Accuracy("multiclass", num_classes=self.num_labels)
        self.acc_val = Accuracy("multiclass", num_classes=self.num_labels)
        self.acc_test = Accuracy("multiclass", num_classes=self.num_labels)
        self.loss_val = MeanMetric()
        self.loss_test = MeanMetric()

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        _, pred_ids = torch.max(logits, dim=1)
        labels = batch["labels"]
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        
        self.acc_train(pred_ids, labels)

        self.log("train_loss_step", loss, prog_bar=True)
        self.log("train_acc_step", self.acc_train, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs["logits"]
        loss = outputs["loss"]
        _, pred_ids = torch.max(logits, dim=1)
        labels = batch["labels"]
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        self.acc_val(pred_ids, labels)
        self.loss_val(loss)
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.acc_val, prog_bar=True)
        self.log("val_loss_epoch", self.loss_val, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        _, pred_ids = torch.max(logits, dim=1)
        labels = batch["labels"]
        labels = labels[0] if len(labels) == 1 else labels.squeeze()

        self.acc_test(pred_ids, labels)
        
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.acc_test, prog_bar=True)
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self(**batch)
        logits = outputs["logits"]
        pred_ids = torch.max(logits, dim=1)
        return {"batch_idx": batch_idx, "outputs": outputs, "pred_ids": pred_ids}

