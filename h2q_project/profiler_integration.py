import torch
import torch.profiler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json


def profile_memory_usage(model, data_loader, criterion, optimizer, num_epochs=1):
    """Profiles memory usage during training, focusing on data loading,
    model computation, and loss calculation.

    Args:
        model: PyTorch model.
        data_loader: PyTorch DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs: Number of epochs to run the profiling for.
    """
    memory_timeline = []

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, # captures input shape,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("data_loading"):
            pass # Placeholder for data loading. The actual loading happens within the loop

        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(data_loader):
                with torch.profiler.record_function("data_loading"):
                    pass # Data is already loaded via data_loader, keep the block

                optimizer.zero_grad()

                with torch.profiler.record_function("model_computation"):
                    outputs = model(inputs)

                with torch.profiler.record_function("loss_calculation"):
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                prof.step()

    # Process the profiler events and extract memory usage
    memory_events = []
    for event in prof.events():
        if event.name in ["data_loading", "model_computation", "loss_calculation"]:
            memory_events.append({
                "name": event.name,
                "device": event.device.type,  # CPU or CUDA
                "total_allocated": event.memory_usage.current if hasattr(event, 'memory_usage') else 0,
                "time": event.start_time_ns # Added timing information
            })

    # Summarize memory usage (sum up across all events of a category)
    memory_summary = {}
    for event in memory_events:
        name = event["name"]
        device = event["device"]
        if name not in memory_summary:
            memory_summary[name] = {"CPU": 0, "CUDA": 0}

        memory_summary[name][device] += event["total_allocated"]


    return memory_summary

if __name__ == '__main__':
    # Example usage:
    input_size = 10
    hidden_size = 5
    output_size = 2
    batch_size = 32
    num_epochs = 2

    # Dummy model
    class SimpleModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    model = SimpleModel(input_size, hidden_size, output_size)
    if torch.cuda.is_available():
        model = model.cuda()

    # Dummy data
    inputs = torch.randn(100, input_size)
    labels = torch.randint(0, output_size, (100,))
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    dataset = TensorDataset(inputs, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    memory_profile = profile_memory_usage(model, data_loader, criterion, optimizer, num_epochs)

    print(json.dumps(memory_profile, indent=4))
