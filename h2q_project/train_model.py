import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from h2q_project.model import SimpleNN  # 假设 SimpleNN 在 model.py 中定义
from h2q_project.dataset import SimpleDataset  # 假设 SimpleDataset 在 dataset.py 中定义
from h2q_project.trainer import Trainer  # Import the Trainer class

# Hyperparameters
input_size = 10
output_size = 5
hidden_size = 20
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_dataset = SimpleDataset(num_samples=100, input_size=input_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = SimpleNN(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop using the Trainer class
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs
)

trainer.train()

print('Finished Training')
