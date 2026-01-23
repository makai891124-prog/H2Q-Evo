import torch

class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % 10 == 0:
                    print (f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
