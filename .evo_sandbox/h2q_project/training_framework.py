import time

class TrainingFramework:
    def __init__(self, model, data_loader, optimizer, reflection_module=None):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.reflection_module = reflection_module # Integrate reflection module

    def train_epoch(self):
        total_loss = 0
        for i, (inputs, labels) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.calculate_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.data_loader)

    def calculate_loss(self, outputs, labels):
        # Dummy loss function, replace with actual loss
        return (outputs - labels).abs().mean()

    def train(self, num_epochs):
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            loss = self.train_epoch()
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Invoke reflection module after each epoch
            if self.reflection_module:
                reflection = self.reflection_module.reflect(loss, epoch_time)
                print(f"Reflection: {reflection}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time:.2f}s")
