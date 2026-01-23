import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from h2q_project.models import AlternativeNN  # Import the alternative model
from h2q_project.trainer import Trainer

# Generate some dummy data
input_size = 20  # Different input size for the alternative model
output_size = 10
batch_size = 64
train_size = 1500
val_size = 300

train_data = torch.randn(train_size, input_size)
train_labels = torch.randint(0, output_size, (train_size,))
val_data = torch.randn(val_size, input_size)
val_labels = torch.randint(0, output_size, (val_size,))

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model, optimizer, and loss function
model = AlternativeNN(input_size, output_size)  # Use the alternative model
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize and train the Trainer
trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, criterion, device)
trainer.train(num_epochs=5)
