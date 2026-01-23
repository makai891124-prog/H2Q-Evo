import torch
import torch.nn as nn
from h2q_project.models.quaternion_layers import QuaternionLinear
from h2q_project.utils.quaternion_utils import quaternion_conjugate, quaternion_product, quaternion_to_rotation_matrix

class GeometryAwareTrainer:
    def __init__(self, model, optimizer, perturbation_magnitude=0.01):
        self.model = model
        self.optimizer = optimizer
        self.perturbation_magnitude = perturbation_magnitude

    def quaternion_perturbation(self, layer):
        """Generates a small quaternion perturbation.

        Args:
            layer: The QuaternionLinear layer to perturb.

        Returns:
            A quaternion perturbation tensor.
        """
        # Generate random noise for the quaternion components
        xi, yi, zi = torch.randn(3, device=layer.weight.device, dtype=layer.weight.dtype)

        # Construct the perturbation quaternion (small angle approximation)
        q_perturbation = torch.cat([torch.ones(1, device=layer.weight.device, dtype=layer.weight.dtype), self.perturbation_magnitude * torch.tensor([xi, yi, zi], device=layer.weight.device, dtype=layer.weight.dtype)], dim=0)
        
        # Normalize the quaternion to ensure it's a unit quaternion
        q_perturbation = q_perturbation / torch.linalg.norm(q_perturbation)
        
        return q_perturbation


    def apply_quaternion_perturbation(self, layer, q_perturbation):
        """Applies the quaternion perturbation to the layer's weight.

        Args:
            layer: The QuaternionLinear layer to perturb.
            q_perturbation: The quaternion perturbation to apply.
        """
        # Get the current quaternion weight
        q_weight = layer.weight

        # Apply the perturbation: q' = q_perturbation * q * conjugate(q_perturbation)
        q_weight_perturbed = quaternion_product(q_perturbation, quaternion_product(q_weight, quaternion_conjugate(q_perturbation)))

        # Update the layer's weight with the perturbed quaternion
        layer.weight.data = q_weight_perturbed

    def train_step(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = nn.CrossEntropyLoss()(output, target)

        # Apply quaternion perturbation to QuaternionLinear layers
        for name, module in self.model.named_modules():
            if isinstance(module, QuaternionLinear):
                # Generate a perturbation
                q_perturbation = self.quaternion_perturbation(module)

                # Apply the perturbation
                self.apply_quaternion_perturbation(module, q_perturbation)

        loss.backward()
        self.optimizer.step()

        return loss.item()
