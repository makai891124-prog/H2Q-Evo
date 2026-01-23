import abc
import torch
from torch.utils.data import DataLoader

class BaseTrainer(abc.ABC):
    """Abstract base class for trainers.

    Defines the structure for training loops, requiring
    implementation of core components like data loading,
    loss calculation, and optimization.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        """Initializes the trainer.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer to use for training.
            device: The device (CPU or GPU) to train on.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    @abc.abstractmethod
    def get_train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set.

        Must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def get_eval_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the evaluation set.

        Must be implemented by subclasses.
        """
        pass


    @abc.abstractmethod
    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the loss for a given batch.

        Must be implemented by subclasses.
        Args:
            batch: A batch of data.

        Returns:
            The computed loss.
        """
        pass

    @abc.abstractmethod
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Performs a single training step.

        This includes forward pass, loss calculation,
        backpropagation, and optimization.
        Must be implemented by subclasses.
        Args:
            batch: A batch of data.

        Returns:
            The computed loss.
        """
        pass

    @abc.abstractmethod
    def evaluation_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Performs a single evaluation step.

        This includes forward pass and loss calculation.
        Must be implemented by subclasses.
        Args:
            batch: A batch of data.

        Returns:
            The computed loss.
        """
        pass

    def train_epoch(self, epoch: int):
        """Trains the model for one epoch.

        Args:
            epoch: The current epoch number.
        """
        self.model.train()
        dataloader = self.get_train_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            loss = self.training_step(batch)

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluates the model on the evaluation set.

        Returns:
            The average loss on the evaluation set.
        """
        self.model.eval()
        dataloader = self.get_eval_dataloader()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(self.device)
            loss = self.evaluation_step(batch)
            total_loss += loss.item()

        return total_loss / len(dataloader)


    def train(self, num_epochs: int):
        """Trains the model for a specified number of epochs.

        Args:
            num_epochs: The number of epochs to train for.
        """
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            eval_loss = self.evaluate()
            print(f'Epoch: {epoch}, Evaluation Loss: {eval_loss}')