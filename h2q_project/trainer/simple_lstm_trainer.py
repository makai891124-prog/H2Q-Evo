import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from h2q_project.trainer.base_trainer import BaseTrainer


class SimpleLSTMTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, train_dataloader: DataLoader, val_dataloader: DataLoader = None, clip: float = 1.0):
        super().__init__(model, optimizer, criterion, device, train_dataloader, val_dataloader)
        self.clip = clip

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = self.model(text, text_lengths).squeeze(1)
            loss = self.criterion(predictions, batch.label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_dataloader)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        epoch_loss = 0
        for i, batch in enumerate(self.val_dataloader):
            text, text_lengths = batch.text
            predictions = self.model(text, text_lengths).squeeze(1)
            loss = self.criterion(predictions, batch.label)
            epoch_loss += loss.item()
        return epoch_loss / len(self.val_dataloader)