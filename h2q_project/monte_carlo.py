import torch

def calculate_pi(samples: int) -> float:
    """Calculates pi using Monte Carlo simulation.

    Args:
        samples: The number of random samples to use.

    Returns:
        An estimate of pi.
    """
    x = torch.rand(samples)
    y = torch.rand(samples)
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * inside.float().mean()
    return pi_estimate

if __name__ == '__main__':
    num_samples = 1000000
    
    # Compile the function using torch.compile
    compiled_calculate_pi = torch.compile(calculate_pi)
    
    pi = compiled_calculate_pi(num_samples)
    print(f"Estimated value of pi: {pi}")