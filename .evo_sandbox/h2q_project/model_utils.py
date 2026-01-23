# h2q_project/model_utils.py
import numpy as np
from h2q_project.settings import DEFAULT_DATA_TYPE

def convert_to_dtype(data):
    if DEFAULT_DATA_TYPE == "float16":
        dtype = np.float16
    elif DEFAULT_DATA_TYPE == "bfloat16":
        # Requires numpy version >= 1.24
        try:
            dtype = np.dtype('float16') #simulate bfloat16, must have numpy installed and version 1.24+
        except TypeError:
            dtype = np.float32 #fallback

        print("BFloat16 not natively supported. Using Float16 instead.")
    else:
        dtype = np.float32

    return data.astype(dtype)


def create_model(input_size, hidden_size, output_size):
    """Creates a simple neural network model."""
    # Example using numpy (replace with your actual model creation logic)
    model = {
        'W1': convert_to_dtype(np.random.randn(input_size, hidden_size)),
        'b1': convert_to_dtype(np.zeros(hidden_size)),
        'W2': convert_to_dtype(np.random.randn(hidden_size, output_size)),
        'b2': convert_to_dtype(np.zeros(output_size))
    }
    return model


