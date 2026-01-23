import time
import psutil
import os


def train(model, data_loader, optimizer, epochs=10):
    """Simple training loop."""
    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s')
                start_time = time.time()

                # Memory monitoring
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 ** 2  # in MB
                print(f"Memory Usage: {memory_usage:.2f} MB")

    print("Training finished!")
