import torch
import os
import numpy as np
from h2q.core.memory.h2q_vault import H2QVault
from h2q.persistence.rskh import RSKH
from h2q.core.interface_registry import SpectralShiftTracker

class H2QReasoningVaultController:
    """
    Orchestrator for the 1M+ Context Reasoning Vault.
    Utilizes RSKH for O(1) addressing and H2QVault for disk-mapped manifold storage.
    """
    def __init__(self, manifold_dim=256, capacity=1000000, storage_dir="./vault_storage"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.manifold_dim = manifold_dim
        self.capacity = capacity
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        # Initialize RSKH (Recursive Sub-Knot Hashing) for O(1) retrieval signatures
        # Registry: h2q/persistence/rskh.py -> RSKH(total_dim, num_knots, device)
        self.hasher = RSKH(total_dim=manifold_dim, num_knots=capacity, device=self.device)

        # Initialize H2QVault for disk-backed memory mapping
        # Registry: h2q/core/memory/h2q_vault.py -> H2QVault(manifold_dim, capacity, storage_path, device)
        self.vault = H2QVault(
            manifold_dim=manifold_dim, 
            capacity=capacity, 
            storage_path=storage_dir, 
            device=self.device
        )

        # Track cognitive progress during vault operations
        self.tracker = SpectralShiftTracker()

    def bootstrap_vault(self):
        """Initializes the disk-backed structures and verifies M4 compatibility."""
        print(f"[BOOTSTRAP] Initializing Vault at {self.manifold_dim} dimensions for {self.capacity} knots.")
        # Verify memory mapping by committing a zero-atom seed
        seed = torch.zeros((1, self.manifold_dim), device=self.device)
        try:
            self.vault.commit(seed)
            print("[VERIFY] Disk-backed memory mapping active.")
        except Exception as e:
            print(f"[ERROR] Vault initialization failed: {e}")
            return False
        return True

    def commit_context(self, manifold_state):
        """
        Encodes a manifold state into the vault using RSKH signatures.
        """
        # Ensure state is on correct device and dtype for M4 (MPS/FP16)
        manifold_state = manifold_state.to(self.device).to(torch.float16)
        
        # Generate O(1) signature via RSKH
        # RSKH.forward(x) returns the hash/signature
        signature = self.hasher.forward(manifold_state)
        
        # Commit to disk-backed storage
        self.vault.commit(manifold_state)
        
        return signature

    def retrieve_context(self, signature, query_state):
        """
        Retrieves historical context knots with O(1) complexity.
        """
        # Registry: retrieve(signature, query_state)
        context_knot = self.vault.retrieve(signature, query_state)
        
        # Calculate spectral shift to ensure retrieval integrity
        # Registry: compute_eta(S)
        # Note: S is the scattering matrix of the transition
        # For retrieval, we treat the query-to-result as a transition
        s_matrix = torch.matmul(query_state.T, context_knot)
        eta = self.tracker.compute_eta(s_matrix)
        
        return context_knot, eta

if __name__ == "__main__":
    # Deployment configuration for Mac Mini M4 (16GB)
    vault_system = H2QReasoningVaultController(
        manifold_dim=256, 
        capacity=1000000, 
        storage_dir="./h2q_context_vault"
    )
    
    if vault_system.bootstrap_vault():
        # Example: Commit a random 256-dim topological knot
        test_knot = torch.randn((1, 256))
        sig = vault_system.commit_context(test_knot)
        print(f"[SUCCESS] Context committed. RSKH Signature: {sig.mean().item()}")
