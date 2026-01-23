import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
from h2q_project.trainer import Trainer # Import the Trainer class

# Define a simple regression dataset
class RegressionDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.X = torch.randn(num_samples, 1)
        self.y = 2 * self.X + 1 + torch.randn(num_samples, 1) * 0.1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def main():
    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create the dataset
    train_dataset = RegressionDataset(num_samples=800)
    val_dataset = RegressionDataset(num_samples=200)

    # Create the model
    model = LinearRegression()

    # Update config for regression specific parameters if necessary
    config['loss_fn'] = 'MSELoss' # Example: force MSELoss, override config

    # Create the Trainer instance
    trainer = Trainer(model, train_dataset, val_dataset, config_path='config.yaml')

    # Train the model
    trainer.train()

if __name__ == '__main__':
    main()
