import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
from h2q_project.trainer import Trainer, load_config
import os

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ModelA_Trainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _create_model(self):
        input_size = self.config['model']['input_size']
        hidden_size = self.config['model']['hidden_size']
        output_size = self.config['model']['output_size']
        return SimpleModel(input_size, hidden_size, output_size)

    def _create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer']['learning_rate'])

    def _create_criterion(self):
        return torch.nn.CrossEntropyLoss()


def create_dummy_data(input_size, num_samples):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, 2, (num_samples,)).long()  # Binary classification
    return X, y


def main():
    config_path = 'config_model_a.yaml'
    config = load_config(config_path)

    # Create dummy data
    input_size = config['model']['input_size']
    num_samples = 1000
    X, y = create_dummy_data(input_size, num_samples)
    dataset = TensorDataset(X, y)

    # Split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['dataloader']['batch_size'], shuffle=False)

    trainer = ModelA_Trainer(config)

    if config.get('load_from_checkpoint'):
      start_epoch, _ = trainer.load_checkpoint(config['load_from_checkpoint'])
      num_epochs = config['training']['num_epochs'] + start_epoch #Adjust for already trained epochs
    else:
        num_epochs = config['training']['num_epochs']


    trainer.train(train_dataloader, val_dataloader, num_epochs)

if __name__ == "__main__":
    main()
