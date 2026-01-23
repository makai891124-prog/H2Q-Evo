import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from h2q_project.trainer import Trainer # Import the Trainer class

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate some dummy data
input_size = 10
hidden_size = 5
output_size = 2
batch_size = 32
num_epochs = 10

# Create dummy data
train_data = torch.randn(100, input_size)
train_labels = torch.randint(0, output_size, (100,))
val_data = torch.randn(50, input_size)
val_labels = torch.randint(0, output_size, (50,))

# Create TensorDatasets
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = SimpleModel(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Instantiate the Trainer
trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, criterion)

# Train the model
trainer.train(num_epochs)
