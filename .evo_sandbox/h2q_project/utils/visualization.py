import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_gradient(gradients, title="Gradient Visualization"):
    """Visualizes the gradient tensor as an image.

    Args:
        gradients (torch.Tensor): Gradient tensor.
        title (str): Title of the plot.
    """
    if gradients is None:
        print("No gradients to visualize.")
        return

    # Convert to numpy and detach from computation graph
    gradients = gradients.detach().numpy()

    # Handle different gradient shapes.  Assume channel first.
    if len(gradients.shape) == 3:
      # Gradients are likely (C, H, W) so transpose for display (H, W, C)
      gradients = np.transpose(gradients, (1, 2, 0))
    elif len(gradients.shape) > 3:
      print(f"Warning: Gradient tensor has shape {gradients.shape}.  Visualization may not be correct.")

    plt.figure()
    plt.imshow(gradients, cmap='viridis')  # You can change the colormap
    plt.title(title)
    plt.colorbar()
    plt.show()
