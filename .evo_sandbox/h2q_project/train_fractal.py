import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from h2q_project.fractal_model import FractalModel
from h2q_project.trainer import Trainer # Import the Trainer class

# Generate synthetic data for fractal training (replace with your actual data)
def generate_fractal_data(num_samples=1000, input_size=10):
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, 1) # Example: Regression task
    return X, y

# Hyperparameters
input_size = 10
learning_rate = 0.001
batch_size = 32
epochs = 10

# Generate data
X, y = generate_fractal_data()

# Create datasets and dataloaders
train_dataset = TensorDataset(X[:800], y[:800])
val_dataset = TensorDataset(X[800:], y[800:])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Model, optimizer, and loss function
model = FractalModel(input_size=input_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Trainer
trainer = Trainer(model, optimizer, loss_fn, device)

# Train the model using the Trainer class
trainer.train(train_dataloader, val_dataloader, epochs)

print("Fractal training complete.")