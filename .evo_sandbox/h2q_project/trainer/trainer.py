import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from typing import Dict, Any

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.device = self._get_device()

        self.model.to(self.device)

    def _get_device(self) -> torch.device:
        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _create_model(self) -> nn.Module:
        # Placeholder:  Replace with your actual model instantiation logic
        # Example: from models import MyModel; return MyModel(**self.config["model_params"])
        raise NotImplementedError("Model creation logic must be implemented.")

    def _create_optimizer(self) -> optim.Optimizer:
        optimizer_config = self.config["optimizer"]
        optimizer_name = optimizer_config["name"]
        optimizer_params = optimizer_config["params"]

        if optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer_name == "SGD":
            return optim.SGD(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_criterion(self) -> nn.Module:
        criterion_name = self.config["criterion"]
        if criterion_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif criterion_name == "MSELoss":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")

    def train_epoch(self, data_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(data_loader.dataset)

    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
        return running_loss / len(data_loader.dataset)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> None:
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config["model_save_path"])
                print("Saved new best model")

    def load_model(self, model_path: str) -> None:
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

def main():
    # Example Usage
    config_path = "config.yaml" # Replace with your config file path

    #Create dummy config.yaml if it doesn't exist
    if not os.path.exists(config_path):
        dummy_config = {
            "device": "cpu",
            "optimizer": {"name": "Adam", "params": {"lr": 0.001}},
            "criterion": "CrossEntropyLoss",
            "model_save_path": "model.pth",
            "epochs": 10 #number of epochs to train for
        }
        with open(config_path, 'w') as f:
            yaml.dump(dummy_config, f)
        print(f"{config_path} not found. Creating a dummy file. Please modify it.")


    config = Trainer.load_config(config_path)
    trainer = Trainer(config)

    # Create dummy data loaders for demonstration
    train_dataset = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(100)]
    val_dataset = [(torch.randn(10), torch.randint(0, 2, (1,)).item()) for _ in range(50)]

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train the model
    trainer.train(train_loader, val_loader, config["epochs"])

if __name__ == "__main__":
    main()