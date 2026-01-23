import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from h2q_project.models import SimpleRegressor  # 确保路径正确
from h2q_project.trainer import Trainer

# Generate some dummy data
X_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randn(100, 1)  # 100 labels, 1 output
X_val = torch.randn(50, 10)
y_val = torch.randn(50, 1)  # validation data

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Instantiate the model
model = SimpleRegressor(input_size=10)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the Trainer
trainer = Trainer(model, optimizer, criterion, device)

# Train the model
trainer.train(train_loader, val_loader, epochs=10)
