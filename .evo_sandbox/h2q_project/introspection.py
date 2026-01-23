import torch
import torch.nn as nn
import torch.profiler
import os
import time

class IntrospectionModule:
    def __init__(self, model, log_dir='introspection_logs'):
        self.model = model
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.forward_hooks = []
        self.backward_hooks = []
        self.profiler = None

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module):
                self.forward_hooks.append(
                    module.register_forward_hook(self._forward_hook(name))
                )
                self.backward_hooks.append(
                    module.register_backward_hook(self._backward_hook(name))
                )

    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        self.forward_hooks = []
        self.backward_hooks = []

    def _forward_hook(self, module_name):
        def forward_hook(module, input, output):
            # Log the output of the layer.
            output_path = os.path.join(self.log_dir, f'{module_name}_output.pt')
            torch.save(output, output_path)

        return forward_hook

    def _backward_hook(self, module_name):
        def backward_hook(module, grad_input, grad_output):
            # Log the gradients of the layer.
            grad_path = os.path.join(self.log_dir, f'{module_name}_grad.pt')
            torch.save(grad_output, grad_path)

        return backward_hook

    def start_profiling(self, record_shapes=True, profile_memory=True, with_stack=True):
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard(self.log_dir),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        )
        self.profiler.start()

    def stop_profiling(self):
        if self.profiler:
            self.profiler.stop()

    def step_profiling(self):
        if self.profiler:
            self.profiler.step()

    def log_resource_usage(self):
        # Log CPU and GPU usage.  Requires psutil and GPUtil.
        try:
            import psutil
            import GPUtil

            cpu_usage = psutil.cpu_percent()
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0  # Assuming single GPU

            log_path = os.path.join(self.log_dir, 'resource_usage.log')
            with open(log_path, 'a') as f:
                f.write(f'{time.time()}: CPU Usage: {cpu_usage}%, GPU Usage: {gpu_usage}%.\n')

        except ImportError:
            print("psutil and GPUtil are required for logging resource usage.  Please install them.")

if __name__ == '__main__':
    # Example Usage
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x

    model = SimpleModel()
    introspection = IntrospectionModule(model, log_dir='introspection_example_logs')
    introspection.register_hooks()

    # Dummy input
    dummy_input = torch.randn(1, 10)

    # Perform a forward and backward pass
    output = model(dummy_input)
    output.mean().backward()

    introspection.remove_hooks()

    # Profiling example
    introspection.start_profiling()
    for _ in range(5):
        model(dummy_input)
        introspection.step_profiling()
    introspection.stop_profiling()

    # Resource usage logging example
    introspection.log_resource_usage()
