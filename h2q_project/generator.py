import torch

def generate_tensor(shape):
    """Generates a random PyTorch tensor with the given shape."""
    return torch.randn(shape)

if __name__ == '__main__':
    # Example usage
    shape = (2, 3, 4)
    tensor = generate_tensor(shape)
    print(f"Generated tensor with shape: {tensor.shape}")
    print(f"Tensor data type: {tensor.dtype}")
    print(tensor)
