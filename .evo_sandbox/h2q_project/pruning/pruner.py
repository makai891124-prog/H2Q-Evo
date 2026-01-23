import torch
import torch.nn as nn
import numpy as np

class Pruner:
    def __init__(self, model, prune_percent=0.5):
        self.model = model
        self.prune_percent = prune_percent

    def prune_weights(self):
        """Prunes the weights of the model based on magnitude."""
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        for module, name in parameters_to_prune:
            self._prune_layer_weights(module, name)

    def _prune_layer_weights(self, module, name):
        """Prunes the weights of a single layer."""
        weight = module.weight.data.cpu().numpy()
        abs_weight = np.abs(weight)
        threshold = np.percentile(abs_weight, self.prune_percent * 100)
        mask = abs_weight > threshold
        mask = torch.from_numpy(mask).to(module.weight.device)

        # Apply mask to the weight
        module.weight.data[mask == False] = 0



if __name__ == '__main__':
    # Example Usage (replace with your actual model)
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleModel()

    # Example: Prune 50% of the weights
    pruner = Pruner(model, prune_percent=0.5)
    pruner.prune_weights()

    # Verify that some weights are now zero.
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: Number of non-zero elements = {torch.count_nonzero(param.data)}")
