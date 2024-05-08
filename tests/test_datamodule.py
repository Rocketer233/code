import pytest
from datamodule import *


@pytest.mark.parametrize(
    "dataset_name, data_path, num_classes", [
        ("QNLI", "../datasets/k_shot/k=16/seed=42/QNLI", 2),
        ("SST2", "../datasets/k_shot/k=16/seed=42/SST2", 2),
        ("MNLI-MATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MATCHED", 3),
        ("MNLI-MISMATCHED", "../datasets/k_shot/k=16/seed=42/MNLI-MISMATCHED", 3),
        ("ENRON-SPAM", "../datasets/k_shot/k=16/seed=42/ENRON-SPAM", 2),
        ("TWEETS-HATE-OFFENSIVE", "../datasets/k_shot/k=16/seed=42/TWEETS-HATE-OFFENSIVE", 3)]
)
def test_glue(model_name, task_name, num_classes):
    train, val, _ = GLUEDataModule(model_name=model_name, task_name=task_name)
    print(train[0])
    assert len(train) == len(val)
    
def test_alpaca(model_name, task_name, num_classes):
    train, val, _ = AlpacaDataModule(model_name=model_name, task_name=task_name)
    print(train[0])
    assert len(train) == len(val)