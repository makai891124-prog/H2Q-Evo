import torch
import torch.nn as nn
import torch.optim as optim
from h2q_project.datasets import SegmentationDataset
from h2q_project.models import UNet
from h2q_project.trainer import Trainer

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Datasets
train_dataset = SegmentationDataset(image_size=64, num_samples=800)
val_dataset = SegmentationDataset(image_size=64, num_samples=200)

# Model
model = UNet(in_channels=3, out_channels=1).to(DEVICE)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Trainer
trainer = Trainer(model, train_dataset, val_dataset, optimizer, criterion, BATCH_SIZE, NUM_EPOCHS, DEVICE)

# Train
trainer.train()
