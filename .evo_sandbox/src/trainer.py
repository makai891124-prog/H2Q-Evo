import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Optional

from src.loss import DRaFTLoss, SequenceCrossEntropyLoss
from src.models.diffusion import DiffusionTransformer
from src.models.transformer import TransformerModel
from src.utils import calculate_metrics
from src.dataset import TextDataset


class TrainerConfig:
    def __init__(self,
                 batch_size: int = 32,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 epochs: int = 10,
                 device: str = 'cuda',
                 num_workers: int = 4,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 ckpt_path: Optional[str] = None,
                 sample_every_n_epoch: int = 1,  # Frequency of sampling
                 num_samples: int = 4,  # Number of samples to generate
                 generation_length: int = 128,  # Length of generated sequences
                 temperature: float = 1.0
                 ):
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.num_workers = num_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.ckpt_path = ckpt_path
        self.sample_every_n_epoch = sample_every_n_epoch
        self.num_samples = num_samples
        self.generation_length = generation_length
        self.temperature = temperature


class Trainer:
    def __init__(self, model: nn.Module, train_dataset, val_dataset, config: TrainerConfig):
        self.model = model.to(config.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        self.step = 0

        self.loss_fn = self._create_loss_fn()

    def _create_loss_fn(self):
        """Override this to define the loss function."""
        raise NotImplementedError

    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            loop = tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{self.config.epochs}')
            for batch in loop:
                # Move data to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass
                loss = self.calculate_loss(batch)

                # Backward pass
                loss.backward()

                # Gradient accumulation and optimization
                if self.step % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update progress bar
                loop.set_postfix(loss=loss.item())
                self.step += 1

            # Validation
            self.validate(epoch)

            # Sampling (conditional)
            if (epoch + 1) % self.config.sample_every_n_epoch == 0:
                self.sample(epoch)

            # Save checkpoint
            if self.config.ckpt_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'step': self.step,
                }, os.path.join(self.config.ckpt_path, f'checkpoint_epoch_{epoch + 1}.pth'))

    def calculate_loss(self, batch):
        """Override this to define how loss is calculated."""
        raise NotImplementedError

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                loss = self.calculate_loss(batch)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        print(f'Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}')

        # Calculate and print metrics (Override this if needed)
        metrics = calculate_metrics(self.model, self.val_dataset, self.config.device, batch_size=self.config.batch_size)
        print(f'Epoch {epoch + 1} Validation Metrics: {metrics}')

    def sample(self, epoch):
        """Override this to implement sampling/generation logic."""
        raise NotImplementedError


class TransformerTrainer(Trainer):
    def __init__(self, model: TransformerModel, train_dataset, val_dataset, config: TrainerConfig):
        super().__init__(model, train_dataset, val_dataset, config)

    def _create_loss_fn(self):
        return SequenceCrossEntropyLoss()

    def calculate_loss(self, batch):
        return self.loss_fn(self.model, batch['input_ids'], batch['attention_mask'], batch['labels'])

    def sample(self, epoch):
         # Get a random sample from the validation dataset to use as a prompt
        sample_idx = torch.randint(0, len(self.val_dataset), (1,)).item()
        prompt = self.val_dataset[sample_idx]['input_ids'].unsqueeze(0).to(self.config.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.config.num_samples):
                generated_sequence = self.model.generate(prompt, max_length=self.config.generation_length, temperature=self.config.temperature)
                generated_text = self.val_dataset.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
                print(f'Generated Text (Sample {i + 1}): {generated_text}')

class DiffusionTrainer(Trainer):
    def __init__(self, model: DiffusionTransformer, train_dataset, val_dataset, config: TrainerConfig):
        super().__init__(model, train_dataset, val_dataset, config)

    def _create_loss_fn(self):
        return DRaFTLoss(self.model)

    def calculate_loss(self, batch):
        return self.loss_fn(batch['input_ids'])

    def sample(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for i in range(self.config.num_samples):
                sampled_sequence = self.model.sample(self.config.generation_length, self.config.temperature, device=self.config.device)
                sampled_text = self.val_dataset.tokenizer.decode(sampled_sequence[0], skip_special_tokens=True)
                print(f'Generated Text (Sample {i + 1}): {sampled_text}')