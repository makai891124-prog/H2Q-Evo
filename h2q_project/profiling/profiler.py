import torch
import torch.profiler
import time


def profile_model(model, input_tensor, num_warmup=3, num_profile=5, profile_memory=True, profile_cpu=True):
    """Profiles a PyTorch model using torch.profiler.

    Args:
        model: The PyTorch model to profile.
        input_tensor: A sample input tensor for the model.
        num_warmup: Number of warmup iterations.
        num_profile: Number of profiling iterations.
        profile_memory: Whether to profile memory usage.
        profile_cpu: Whether to profile CPU usage.
    """
    wait = 1
    warmup = num_warmup
    active = num_profile
    repeat = 1

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        profile_memory=profile_memory, 
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for step in range(wait + warmup + active):
                model(input_tensor)
                prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == '__main__':
    # Example usage:
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1000, 100)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x

    model = SimpleModel()
    input_tensor = torch.randn(1, 1000)

    profile_model(model, input_tensor)
