import torch
import os
from h2q.persistence.rskh import RSKH
from h2q.core.serialization.manifold_snapshot import ManifoldSnapshot, RSKHEncoder

class H2QVault:
    """
    H2Q-Vault: Persistent memory layer for O(1) retrieval of context knots.
    Integrates Recursive Sub-Knot Hashing (RSKH) with ManifoldSnapshot storage.
    Optimized for Mac Mini M4 (MPS) with 16GB RAM constraints.
    """
    def __init__(self, manifold_dim=256, capacity=1000000, storage_path="./vault_store", device="mps"):
        self.manifold_dim = manifold_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # RSKH Engine: Manages the recursive hashing of the quaternionic manifold trajectory.
        # Architecture: 256-dimensional manifold (64-knot clusters).
        self.rskh_engine = RSKH(total_dim=manifold_dim, num_knots=64, device=self.device)
        
        # RSKH Encoder: Converts high-dimensional knots into stable, storage-friendly signatures.
        self.sig_encoder = RSKHEncoder(input_dim=manifold_dim, seed=42)
        
        # Manifold Snapshot: Handles persistence and O(1) retrieval from disk/memory.
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
            
        self.snapshot_manager = ManifoldSnapshot(
            storage_path=storage_path, 
            capacity=capacity, 
            dim=manifold_dim
        )

    def commit(self, manifold_state):
        """
        Hashes the current manifold state and commits it to the vault.
        Args:
            manifold_state (torch.Tensor): The current 256-dim quaternionic knot.
        Returns:
            str: The generated signature for the context.
        """
        # Ensure tensor is on the correct device and shape
        if manifold_state.device != self.device:
            manifold_state = manifold_state.to(self.device)
            
        if manifold_state.dim() == 1:
            manifold_state = manifold_state.unsqueeze(0)
        
        # 1. Update the recursive hash state with the new knot
        # This ensures the signature represents the entire context history (geodesic flow)
        hash_projection = self.rskh_engine.forward(manifold_state)
        
        # 2. Generate a stable signature from the hash projection
        signature = self.sig_encoder.generate_signature(hash_projection)
        
        # 3. Commit the knot to persistent storage
        # We store the state indexed by the recursive signature to allow O(1) retrieval
        self.snapshot_manager.commit_knot(signature, manifold_state.squeeze(0))
        
        return signature

    def retrieve(self, signature=None, query_state=None):
        """
        Retrieves a manifold state from the vault in O(1) time.
        Args:
            signature (str, optional): Direct key for lookup.
            query_state (torch.Tensor, optional): State to hash for lookup.
        Returns:
            torch.Tensor: The retrieved quaternionic knot.
        """
        if signature is None and query_state is not None:
            if query_state.dim() == 1:
                query_state = query_state.unsqueeze(0)
            # Calculate signature from query state using the current RSKH context
            hash_projection = self.rskh_engine.forward(query_state)
            signature = self.sig_encoder.generate_signature(hash_projection)
        
        if signature is None:
            raise ValueError("H2Q-Vault Error: Must provide either a signature or a query_state for retrieval.")
            
        return self.snapshot_manager.retrieve_knot(signature)

    def get_current_context_hash(self):
        """
        Retrieves the current state of the recursive hash as a signature.
        """
        state = self.rskh_engine.retrieve_state()
        return self.sig_encoder.generate_signature(state)

    def reset_context(self):
        """
        Resets the recursive hashing engine for a new sequence.
        """
        self.rskh_engine = RSKH(total_dim=self.manifold_dim, num_knots=64, device=self.device)
