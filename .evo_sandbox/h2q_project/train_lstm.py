import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from h2q_project.trainer import Trainer

# Define a simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Generate some dummy data
def generate_dummy_data(num_samples=1000, sequence_length=20, input_size=10):
    X = torch.randn(num_samples, sequence_length, input_size)
    y = torch.randint(0, 5, (num_samples,))
    return X, y


def main():
    # Hyperparameters
    input_size = 10
    hidden_size = 32
    output_size = 5
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate dummy data
    X, y = generate_dummy_data()

    # Split into training and validation sets
    train_size = int(0.8 * len(X))
    val_size = len(X) - train_size
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = SimpleLSTM(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize Trainer
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, criterion, device)

    # Train the model
    trainer.train(epochs)

if __name__ == "__main__":
    main()
