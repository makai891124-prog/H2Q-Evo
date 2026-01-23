import torch
import torchvision
import torchvision.transforms as transforms
from h2q_project.models import FashionMNISTClassifier
from h2q_project.trainer import Trainer

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the FashionMNIST training dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Instantiate the model
model = FashionMNISTClassifier()

# Define training parameters
batch_size = 64
learning_rate = 0.001
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the Trainer class
trainer = Trainer(model, train_dataset, batch_size, learning_rate, epochs, device)

# Train the model
trainer.train()
