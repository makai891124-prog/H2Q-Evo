from h2q_project.training.self_reflection import TrainingMonitor

class Trainer:
    def __init__(self, model, optimizer, loss_fn, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.monitor = monitor if monitor else TrainingMonitor()

    def train_step(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        
        # Log gradient norm
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        self.monitor.log_gradient_norm(grad_norm)

        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            for i, (data, target) in enumerate(train_loader):
                loss = self.train_step(data, target)
                self.monitor.log_loss(loss)

                if i % 10 == 0:
                    status = self.monitor.get_status()
                    if any(status.values()):
                        print(f"Epoch: {epoch}, Batch: {i}, Status: {status}")
                        action = self.monitor.suggest_action()
                        print(f"Suggested action: {action}")
                        # Implement logic to automatically adjust training parameters here
                        # For example, you could modify the learning rate of the optimizer

            print(f"Epoch {epoch} complete")
