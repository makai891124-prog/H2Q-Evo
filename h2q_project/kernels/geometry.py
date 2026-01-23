import torch
import torch.profiler


def compute_distances(x, y):
    """Computes pairwise distances between two sets of points."""
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, torch.max(dist))


def compute_gram_matrix(x, y, kernel_func):
    """Computes the Gram matrix using a given kernel function."""
    return kernel_func(x, y)


def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian kernel function."""
    distances = compute_distances(x, y)
    return torch.exp(-distances / (2 * sigma**2))


def laplacian_kernel(x, y, sigma=1.0):
    """Laplacian kernel function."""
    distances = compute_distances(x, y)
    return torch.exp(-torch.sqrt(distances) / sigma)


if __name__ == '__main__':
    # Example Usage and Profiling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(100, 32, device=device)
    y = torch.randn(120, 32, device=device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("gaussian_kernel_forward"):    
            gram_matrix = compute_gram_matrix(x, y, gaussian_kernel)

    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # Optional: Save profiling results to file
    # prof.export_chrome_trace("trace.json")

    print("Gram Matrix Shape:", gram_matrix.shape)
