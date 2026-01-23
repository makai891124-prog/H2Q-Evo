from h2q_project.utils.logger import Logger

# Assume you have a training loop like this:

def train(model, data_loader, optimizer, epochs=10):
    logger = Logger()
    for epoch in range(epochs):
        # Perform training step
        loss, accuracy = train_step(model, data_loader, optimizer)

        # Log the metrics
        logger.log({'epoch': epoch + 1, 'loss': loss, 'accuracy': accuracy})

    # Save the log after training
    logger.save_log()

def train_step(model, data_loader, optimizer):
    # Dummy implementation - replace with your actual training step
    return 0.1, 0.9 # dummy loss and accuracy


if __name__ == '__main__':
    # Example usage:
    class DummyModel:
        pass

    class DummyDataLoader:
        pass

    class DummyOptimizer:
        pass

    model = DummyModel()
    data_loader = DummyDataLoader()
    optimizer = DummyOptimizer()

    train(model, data_loader, optimizer)
