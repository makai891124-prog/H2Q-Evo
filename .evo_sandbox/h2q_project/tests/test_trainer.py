import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from h2q_project.trainer import Trainer  # Assuming trainer.py is in the same directory


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DummyModel().to(self.device)

        # Create dummy data
        self.train_data = torch.randn(100, 10).to(self.device)
        self.train_labels = torch.randn(100, 5).to(self.device)
        self.val_data = torch.randn(50, 10).to(self.device)
        self.val_labels = torch.randn(50, 5).to(self.device)

        self.train_dataset = TensorDataset(self.train_data, self.train_labels)
        self.val_dataset = TensorDataset(self.val_data, self.val_labels)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32)
        self.epochs = 2

    def test_trainer_with_mse_and_adam(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        trainer = Trainer(self.model, self.device, self.train_loader, self.val_loader, optimizer, loss_fn, self.epochs)
        trainer.train()

    def test_trainer_with_l1_and_sgd(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = nn.L1Loss()
        trainer = Trainer(self.model, self.device, self.train_loader, self.val_loader, optimizer, loss_fn, self.epochs)
        trainer.train()

    def test_nan_detection(self):
        # Create a model that will produce NaNs
        class NaNModel(nn.Module):
            def __init__(self):
                super(NaNModel, self).__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                # Force NaN output.  This is just a dummy example.
                output = torch.exp(x)
                output[0,0] = float('nan')
                return output

        nan_model = NaNModel().to(self.device)
        optimizer = optim.Adam(nan_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # Create dummy data for the NaN model
        nan_train_data = torch.randn(10, 10).to(self.device)
        nan_train_labels = torch.randn(10, 1).to(self.device)
        nan_val_data = torch.randn(5, 10).to(self.device)
        nan_val_labels = torch.randn(5, 1).to(self.device)

        nan_train_dataset = TensorDataset(nan_train_data, nan_train_labels)
        nan_val_dataset = TensorDataset(nan_val_data, nan_val_labels)

        nan_train_loader = DataLoader(nan_train_dataset, batch_size=2)
        nan_val_loader = DataLoader(nan_val_dataset, batch_size=2)

        trainer = Trainer(nan_model, self.device, nan_train_loader, nan_val_loader, optimizer, loss_fn, self.epochs)

        # Expect training to stop due to NaN loss
        trainer.train()


if __name__ == '__main__':
    unittest.main()
