import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from h2q_project.models import SimpleNN
from h2q_project.trainer import Trainer

# Hyperparameters
batch_size = 64
learning_rate = 0.01
momentum = 0.5
epochs = 10
log_interval = 100

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std deviation for MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = SimpleNN().to(device)

# Optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

# Trainer
save_dir = './checkpoints'
trainer = Trainer(model, train_loader, test_loader, optimizer, criterion, device, log_interval, save_dir)

# Training loop
trainer.train(epochs)
