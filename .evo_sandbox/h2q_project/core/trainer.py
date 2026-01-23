import time
import random

class Trainer:
    def __init__(self, model, optimizer, loss_fn, data_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.config = config
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience = config.get('patience', 10) # Default patience value
        self.counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for i, (inputs, targets) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def should_stop(self):
        return self.counter >= self.patience

    def run(self):
        import torch
        start_time = time.time()

        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            train_loss = self.train_epoch()
            eval_loss = self.evaluate(self.data_loader.valid_loader)

            print(f"Epoch {epoch+1}/{self.config['epochs']}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

            # Simple Self-Reflection Module
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.counter = 0

                # Save the best model (implementation depends on project structure)
                torch.save(self.model.state_dict(), 'best_model.pth')  # Example: saving to a file
                print("Best model saved.")
            else:
                self.counter += 1
                print(f"Patience: {self.counter}/{self.patience}")                # Hyperparameter Adjustment (Example: Reduce learning rate if specified)
                if self.config.get('adjust_lr', False) and self.counter >= self.patience / 2:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5  # Reduce learning rate by half
                    print("Learning rate reduced.")

            # Early Stopping
            if self.should_stop():
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Total training time: {training_time:.2f} seconds")



# Example Usage (assuming data_loader is an instance with train_loader and valid_loader)
if __name__ == '__main__':
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    # Define a dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return torch.randn(10), torch.randn(5) # Example: 10 input features, 5 output features

    # Create dummy data loaders
    train_dataset = DummyDataset(100)
    valid_dataset = DummyDataset(50)

    train_loader = DataLoader(train_dataset, batch_size=32)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    class DataLoaders:
        def __init__(self, train_loader, valid_loader):
            self.train_loader = train_loader
            self.valid_loader = valid_loader

    data_loader = DataLoaders(train_loader, valid_loader)

    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Define training configuration
    config = {
        'epochs': 10,
        'patience': 3,  # Early stopping patience
        'adjust_lr': True # Enable learning rate adjustment
    }

    # Create and run the trainer
    trainer = Trainer(model, optimizer, loss_fn, data_loader, config)
    trainer.run()
