import torch

data_dir='./data'
batch_size=32
num_classes=3
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs=100
learning_rate=1e-4
weight_decay=1e-4