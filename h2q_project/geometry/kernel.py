import torch
import time

class GeometryKernel:
    def __init__(self, device='cpu'):
        self.device = device

    def compute_interaction(self, sources, targets, charges):
        """Computes the interaction between sources and targets.

        Args:
            sources (torch.Tensor): Source locations (N x 3).
            targets (torch.Tensor): Target locations (M x 3).
            charges (torch.Tensor): Charges at source locations (N x 1).

        Returns:
            torch.Tensor: Potentials at target locations (M x 1).
        """
        # Profile the interaction computation
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # Ensure tensors are on the correct device
            sources = sources.to(self.device)
            targets = targets.to(self.device)
            charges = charges.to(self.device)

            # Compute the distance matrix using optimized kernel
            distance_matrix = self._compute_distance_matrix(sources, targets)

            # Compute the interaction potentials
            potentials = self._compute_potentials(distance_matrix, charges)

        # Print profiler summary
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        print(prof.memory_profile())
        # Export trace for further analysis (optional)
        # prof.export_chrome_trace("trace.json")

        return potentials

    def _compute_distance_matrix(self, sources, targets):
        """Computes the distance matrix between sources and targets.

        This implementation uses explicit loops for clarity.  Can be further optimized
        using vectorized operations and tiling.

        Args:
            sources (torch.Tensor): Source locations (N x 3).
            targets (torch.Tensor): Target locations (M x 3).

        Returns:
            torch.Tensor: Distance matrix (M x N).
        """
        num_targets = targets.shape[0]
        num_sources = sources.shape[0]
        distance_matrix = torch.zeros((num_targets, num_sources), device=self.device)

        for i in range(num_targets):
            for j in range(num_sources):
                distance_matrix[i, j] = torch.sqrt(torch.sum((targets[i, :] - sources[j, :])**2))

        return distance_matrix

    def _compute_potentials(self, distance_matrix, charges):
        """Computes the potentials at target locations.

        Args:
            distance_matrix (torch.Tensor): Distance matrix (M x N).
            charges (torch.Tensor): Charges at source locations (N x 1).

        Returns:
            torch.Tensor: Potentials at target locations (M x 1).
        """
        # Simple inverse distance weighting (can be replaced with more sophisticated kernels)
        potentials = torch.sum(charges.T / (distance_matrix + 1e-6), dim=1, keepdim=True)
        return potentials


if __name__ == '__main__':
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kernel = GeometryKernel(device=device)

    num_sources = 100
    num_targets = 50

    sources = torch.randn(num_sources, 3)
    targets = torch.randn(num_targets, 3)
    charges = torch.randn(num_sources, 1)

    start_time = time.time()
    potentials = kernel.compute_interaction(sources, targets, charges)
    end_time = time.time()

    print("Potentials shape:", potentials.shape)
    print(f"Computation time: {end_time - start_time:.4f} seconds")
