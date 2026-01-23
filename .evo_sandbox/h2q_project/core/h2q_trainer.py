class H2QTrainer:
    """Base class for H2Q trainers."""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, data, labels):
        """Performs a single training step.

        Args:
            data (torch.Tensor): Input data.
            labels (torch.Tensor): Target labels.

        Returns:
            float: Loss value.
        """
        raise NotImplementedError

    def train(self, data_loader, epochs):
        """Trains the model.

        Args:
            data_loader (torch.utils.data.DataLoader): Training data loader.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            for data, labels in data_loader:
                loss = self.train_step(data, labels)
                print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')


class SimpleTrainer(H2QTrainer):
    """A simple trainer for H2Q models, assuming a PyTorch-like model.
    """
    def __init__(self, model, optimizer, loss_fn):
        super().__init__(model, optimizer)
        self.loss_fn = loss_fn

    def train_step(self, data, labels):
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()