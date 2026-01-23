import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os

from h2q_project.trainer.base_trainer import BaseTrainer

class ImageClassificationTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, epochs: int, save_dir: str, logger: logging.Logger):
        super().__init__(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs, logger)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _calculate_loss(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss

    def _save_checkpoint(self, epoch: int, best_val_loss: float):
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)