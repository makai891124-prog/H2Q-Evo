import torch
import torch.nn as nn
from h2q.core.generation import H2QAutoregressiveGenerator
from h2q.core.guards.holomorphic_guard_middleware import HolomorphicGuardMiddleware
from h2q.core.interface_registry import get_canonical_dde

class ProductionLogicalGenerator(nn.Module):
    """
    H2Q Production Generator with Holomorphic Guard Integration.
    Performs real-time logical pruning based on Fueter curvature (Df).
    """
    def __init__(self, dim=256, vocab_size=1024, threshold=0.05, device="mps"):
        super().__init__()
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.threshold = threshold

        # 1. Initialize the Base Generator
        # Note: We ensure DDE initialization inside the generator matches the registry
        # to avoid the 'unexpected keyword argument dim' error.
        self.generator = H2QAutoregressiveGenerator(
            dim=latent_dim, 
            vocab_size=vocab_size, 
            device=self.device
        )

        # 2. Initialize Holomorphic Guard Middleware
        # healing_factor=0.1 allows for slight manifold correction without total collapse
        self.guard = HolomorphicGuardMiddleware(
            dim=latent_dim,
            threshold=threshold,
            healing_factor=0.1,
            device=self.device
        )

        # 3. Attach Guard to the Generator's reasoning core
        # We target the hidden state transition where logical hallucinations (topological tears) occur.
        self.guard.attach(self.generator, "generate_step")

    def generate_with_pruning(self, initial_state, max_steps=50, top_k=5):
        """
        Generates a sequence while pruning branches that exceed the Fueter curvature threshold.
        """
        self.generator.eval()
        results = []
        current_state = initial_state.to(self.device)
        prev_hidden = torch.zeros_like(current_state)

        print(f"[M24-CW] Starting Logical Generation. Threshold: {self.threshold}")

        for step in range(max_steps):
            # Perform generation step
            # The guard's forward_hook will automatically compute Fueter residuals
            logits, next_hidden = self.generator.generate_step(current_state, prev_hidden)
            
            # Calculate Fueter Residual (Df) for the current transition
            # Df measures the deviation from the Quaternionic Cauchy-Riemann equations
            residual = self.guard.compute_fueter_residual(next_hidden)
            max_residual = residual.max().item()

            # LOGICAL PRUNING LOGIC
            if max_residual > self.threshold:
                print(f"[!] Step {step}: Logical Hallucination Detected (Df={max_residual:.4f}). Pruning branch.")
                
                # Option A: Neutralize the tear and continue (Elastic Extension)
                # next_hidden = self.guard.neutralize_tear(next_hidden, residual)
                
                # Option B: Hard Pruning (Rigid Construction)
                # If the logic is too curved, the manifold is unstable; we terminate this sequence branch.
                break

            # Sampling with Top-K
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            results.append(next_token.item())
            prev_hidden = next_hidden
            current_state = next_hidden # In H2Q, hidden state is the manifold coordinate

        return results

if __name__ == "__main__":
    # Compatibility Check for Mac Mini M4
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Initialize System
    model = ProductionLogicalGenerator(dim=256, vocab_size=5000, threshold=0.05, device=device)
    
    # Mock Seed Atom (S³ Manifold Coordinate)
    seed = torch.randn(1, 256)
    
    # Execute Generation
    sequence = model.generate_with_pruning(seed, max_steps=20)
    print(f"Generated Sequence: {sequence}")
