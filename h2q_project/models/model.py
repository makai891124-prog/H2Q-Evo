import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def visualize_features(self, x, target_class):
        """Visualizes the features learned by the model using gradients.

        Args:
            x (torch.Tensor): Input tensor of shape (1, input_size).
            target_class (int): The target class to visualize.

        Returns:
            torch.Tensor: Gradients of the target class output with respect to the input.
                          Returns None if visualization fails.
        """
        x.requires_grad_(True)
        output = self.forward(x)

        # Ensure target_class is within the valid range
        if target_class < 0 or target_class >= output.shape[1]:
            print(f"Error: target_class {target_class} is out of bounds (0 to {output.shape[1]-1})")
            return None

        target_output = output[0, target_class]  # Access the target class's output score

        # Calculate gradients
        target_output.backward()
        gradients = x.grad

        return gradients
