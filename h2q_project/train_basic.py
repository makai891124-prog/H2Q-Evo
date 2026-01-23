import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from h2q_project.trainer import Trainer

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.data = torch.randn(length, 10)
        self.labels = torch.randint(0, 2, (length,))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Dummy Model
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Hyperparameters (can be overridden by config)
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Datasets
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)

    # Model
    model = DummyModel()

    # Configuration for the trainer
    config = {
        'optimizer': {
            'name': 'Adam',
            'params': {'lr': learning_rate}
        },
        'lr_scheduler': {
            'name': 'StepLR',
            'params': {'step_size': 5, 'gamma': 0.1}
        },
        'loss_fn': 'CrossEntropyLoss',
        'batch_size': batch_size,
        'epochs': num_epochs
    }

    # Trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)

    # Train
    trainer.train()