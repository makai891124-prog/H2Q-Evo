import numpy as np

def tensor_add(tensor1, tensor2):
    # Element-wise addition of two tensors
    return np.add(tensor1, tensor2)

def tensor_multiply(tensor1, tensor2):
    # Element-wise multiplication of two tensors
    return np.multiply(tensor1, tensor2)

def tensor_matmul(tensor1, tensor2):
    # Matrix multiplication of two tensors
    return np.matmul(tensor1, tensor2)

def tensor_transpose(tensor):
    # Transpose a tensor
    return np.transpose(tensor)

def tensor_reshape(tensor, new_shape):
    # Reshape a tensor to a new shape
    return np.reshape(tensor, new_shape)

def tensor_sum(tensor, axis=None):
    # Sum of tensor elements along a given axis
    return np.sum(tensor, axis=axis)

def create_tensor(shape, fill_value=0):
    # Creates a tensor of a specified shape and fills it with a specified value
    return np.full(shape, fill_value)
