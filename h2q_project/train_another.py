import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from h2q_project.models.simple_linear import SimpleLinear  # Assuming simple_linear is in models directory
from h2q_project.trainers.base_trainer import BaseTrainer

# Generate some dummy data
x_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
x_test = torch.randn(50, 10)
y_test = torch.randn(50, 1)

# Create TensorDatasets
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
model = SimpleLinear(input_size=10, output_size=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Determine the device to use (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 15
trainer = BaseTrainer(model, optimizer, criterion, device)

for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    test_loss, test_accuracy = trainer.evaluate(test_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

print('Finished Training')
