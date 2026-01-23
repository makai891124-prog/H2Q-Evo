import logging
import numpy as np

logging.basicConfig(level=logging.INFO, filename='training_log.log')

class TrainingModule:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        # Self-reflection module
        self.reflect(loss)

        self.optimizer.step()
        return loss.item()

    def reflect(self, loss):
        # Log loss
        logging.info(f'Loss: {loss.item()}')

        # Check for vanishing/exploding gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.cpu().numpy())
                logging.info(f'Gradient norm for {name}: {grad_norm}')

                if grad_norm > 10: # Example threshold
                    logging.warning(f'Possible exploding gradient detected for {name}')
                    logging.warning('Consider gradient clipping.')
                elif grad_norm < 1e-6: # Example threshold
                    logging.warning(f'Possible vanishing gradient detected for {name}')
                    logging.warning('Consider different activation functions or initialization.')


    def train_epoch(self, dataloader):
        total_loss = 0
        for inputs, targets in dataloader:
            loss = self.train_step(inputs, targets)
            total_loss += loss
        return total_loss / len(dataloader)
