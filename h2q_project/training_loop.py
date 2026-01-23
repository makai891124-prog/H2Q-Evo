import time
import numpy as np
from h2q_project.geometry_kernel import GeometryKernel
from h2q_project.performance_monitor import PerformanceMonitor

class TrainingLoop:
    def __init__(self, model, data_loader, optimizer, hyperparams):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.hyperparams = hyperparams
        self.geometry_kernel = GeometryKernel()
        self.performance_monitor = PerformanceMonitor()

    def train_step(self, data, labels):
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.model.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate_step(self, data, labels):
        outputs = self.model(data)
        loss = self.model.loss_function(outputs, labels)
        return loss.item()

    def adjust_hyperparams(self, performance_metric):
        # Simple self-reflection: Adjust learning rate based on performance.
        # This is a placeholder; a more sophisticated approach is needed in practice.
        if performance_metric > self.hyperparams['performance_threshold']:
            self.hyperparams['learning_rate'] *= 0.9  # Reduce learning rate
            print(f"Adjusting learning rate to {self.hyperparams['learning_rate']}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.hyperparams['learning_rate']
        elif performance_metric < self.hyperparams['performance_lower_threshold']:
            self.hyperparams['learning_rate'] *= 1.1
            print(f"Adjusting learning rate to {self.hyperparams['learning_rate']}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.hyperparams['learning_rate']
        return self.hyperparams

    def run(self, num_epochs):
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for data, labels in self.data_loader:
                loss = self.train_step(data, labels)
                epoch_loss += loss
                num_batches += 1

            epoch_loss /= num_batches
            validation_loss = 0.0
            num_val_batches = 0
            for data, labels in self.data_loader:
                val_loss = self.validate_step(data, labels)
                validation_loss += val_loss
                num_val_batches += 1

            validation_loss /= num_val_batches


            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}, Time: {epoch_duration:.2f}s")
            # Track performance metrics
            self.performance_monitor.track_metric("loss", epoch_loss)
            self.performance_monitor.track_metric("validation_loss", validation_loss)
            performance_metric = self.geometry_kernel.calculate_metric(epoch_loss, validation_loss)

            # Self-reflection: Adjust hyperparameters based on performance
            self.hyperparams = self.adjust_hyperparams(performance_metric)


