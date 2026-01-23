import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from h2q_project.data_loader import CustomDataset
from h2q_project.models.generator import SimpleGenerator
from h2q_project.trainers.base_trainer import BaseTrainer

class GeneratorTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.model = SimpleGenerator(config['input_size'], config['hidden_size'], config['output_size'])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.train_dataset = CustomDataset(config['train_data_path'])
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_dataset = CustomDataset(config['val_data_path'])
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False)

    def train(self):
        for epoch in range(self.config['epochs']):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}')
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        print(f'Validation Loss: {val_loss / len(self.val_loader)}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f'Model loaded from {path}')


if __name__ == '__main__':
    config = {
        'input_size': 100,
        'hidden_size': 50,
        'output_size': 1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'train_data_path': 'data/train.csv',
        'val_data_path': 'data/val.csv',
        'model_save_path': 'models/generator.pth'
    }

    trainer = GeneratorTrainer(config)
    trainer.train()
    trainer.save_model(config['model_save_path'])
