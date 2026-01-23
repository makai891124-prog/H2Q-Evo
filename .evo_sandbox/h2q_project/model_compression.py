import torch
import torch.nn as nn


def quantize_model(model, num_bits=8):
    """Quantizes the weights of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to quantize.
        num_bits (int): The number of bits to use for quantization (e.g., 8 for int8).

    Returns:
        nn.Module: The quantized PyTorch model.
    """

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Quantize weights
            with torch.no_grad():  # Disable gradient calculation during quantization
                # Calculate quantization parameters (scale and zero point)
                min_val = torch.min(module.weight)
                max_val = torch.max(module.weight)
                q_min = 0
                q_max = 2**num_bits - 1

                scale = (max_val - min_val) / (q_max - q_min)
                zero_point = q_min - torch.round(min_val / scale)
                zero_point = torch.clamp(zero_point, q_min, q_max)

                # Quantize the weights
                quantized_weights = torch.round(module.weight / scale + zero_point)
                quantized_weights = torch.clamp(quantized_weights, q_min, q_max)

                # Dequantize the weights
                dequantized_weights = (quantized_weights - zero_point) * scale

                # Replace original weights with dequantized weights
                module.weight.data = dequantized_weights.data

    return model


def prune_model(model, pruning_percentage=0.5):
    """Prunes the weights of a PyTorch model based on magnitude.

    Args:
        model (nn.Module): The PyTorch model to prune.
        pruning_percentage (float): The percentage of weights to prune (e.g., 0.5 for 50%).

    Returns:
        nn.Module: The pruned PyTorch model.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                # Calculate the pruning threshold
                weight_abs = torch.abs(module.weight)
                threshold = torch.quantile(weight_abs.flatten(), pruning_percentage)

                # Create a mask for pruning
                mask = weight_abs > threshold

                # Apply the mask to prune the weights
                module.weight.data[~mask] = 0

    return model


if __name__ == '__main__':
    # Example Usage (replace with your actual model)
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 5)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    model = SimpleModel()
    print("Original Model:", model)

    # Quantize the model (example)
    quantized_model = quantize_model(model)
    print("Quantized Model:", quantized_model)

    # Prune the model (example)
    pruned_model = prune_model(model)
    print("Pruned Model:", pruned_model)