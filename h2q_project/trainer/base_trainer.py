import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class BaseTrainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: str, epochs: int):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs

        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for i, data in enumerate(self.train_dataloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_dataloader)

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_dataloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
        return running_loss / len(self.val_dataloader)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
