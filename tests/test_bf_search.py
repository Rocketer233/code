import csv
from bf_search import *

if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    tasks = [['mnli', 3]]
    
    with open("bfsearch_groupings.csv", "w", newline='') as file:
        writer = csv.writer(file)
        for task_name, num_labels in tasks:
            writer.writerows(bf_search(model_name, task_name, num_labels))
    
    grouping = [[[i] for i in range(num_heads)]] * num_layers
    groupings = [({"k": grouping, "v": grouping}, 0.5)] * 4
    
    with open("bfsearch_groupings.csv", "w", newline='') as file:
        writer = csv.writer(file)
        for i in range(3):
            t = [[i]]
            writer.writerows(groupings)