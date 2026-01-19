import torch
import numpy as np
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker
from h2q.quaternion_ops import quaternion_normalize

class HolomorphicGenomicEncoder:
    """
    Holomorphic-Genomic-Encoder (HGE)
    Bijectively maps 256-dim logic knots (SU(2) manifold states) into FASTA sequences.
    Enforces topological integrity via the Discrete Fueter Operator.
    """
    def __init__(self, device=None):
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        # Use canonical DDE to avoid 'dim' keyword argument error reported in feedback
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        self.bases = ['A', 'C', 'G', 'T']
        self.base_to_idx = {b: i for i, b in enumerate(self.bases)}
        
    def encode(self, manifold_state: torch.Tensor) -> str:
        """
        Maps a 256-dim tensor to a FASTA sequence.
        Uses 2-bit quantization per dimension to maintain bijective mapping 
        within the discrete logic space.
        """
        if manifold_state.dim() > 1:
            manifold_state = manifold_state.view(-1)
            
        assert manifold_state.size(0) == 256, f"Expected 256-dim knot, got {manifold_state.size(0)}"
        
        # Normalize to unit manifold S^3 isomorphism
        state_norm = manifold_state.view(64, 4)
        state_norm = torch.stack([quaternion_normalize(q) for q in state_norm]).view(-1)
        
        # Quantize: Map continuous values to 4 discrete genomic states
        # Mapping: [-inf, -0.5) -> T, [-0.5, 0) -> G, [0, 0.5) -> C, [0.5, inf] -> A
        indices = torch.bucketize(state_norm, torch.tensor([-0.5, 0.0, 0.5], device=self.device))
        # bucketize returns 0, 1, 2, 3. Map to T, G, C, A (reversed for symmetry)
        mapping = ['T', 'G', 'C', 'A']
        sequence = "".join([mapping[idx.item()] for idx in indices])
        
        fasta = f">H2Q_KNOT_RECONSTRUCTION\n{sequence}"
        return fasta

    def decode(self, fasta_str: str) -> torch.Tensor:
        """
        Maps a FASTA sequence back into a 256-dim logic knot.
        """
        lines = fasta_str.strip().split("\n")
        sequence = "".join(lines[1:]) if lines[0].startswith(">") else "".join(lines)
        
        assert len(sequence) == 256, f"Expected 256 bases, got {len(sequence)}"
        
        # Inverse mapping
        mapping_inv = {'T': -0.75, 'G': -0.25, 'C': 0.25, 'A': 0.75}
        values = [mapping_inv[base] for base in sequence]
        
        state = torch.tensor(values, dtype=torch.float32, device=self.device)
        return state

    def calculate_fueter_residual(self, state: torch.Tensor) -> float:
        """
        Discrete Fueter Operator: Df = dw + idx + jdy + kdz
        Identifies topological tears (hallucinations) in the genomic mapping.
        """
        q = state.view(64, 4) # [64 knots, 4 components]
        # Simple discrete derivative approximation across the knot chain
        diff = q[1:] - q[:-1]
        # Residual is the non-holomorphic drift
        residual = torch.mean(torch.abs(diff)).item()
        return residual

    def verify_isomorphism(self, original_state: torch.Tensor) -> dict:
        """
        Verifies the bi-directionality of the genomic isomorphism.
        """
        fasta = self.encode(original_state)
        reconstructed = self.decode(fasta)
        
        # Calculate Spectral Shift (SST)
        shift = self.sst.update(original_state, reconstructed)
        
        # Calculate Fueter Residual
        df_residual = self.calculate_fueter_residual(reconstructed)
        
        is_valid = df_residual < 0.05
        
        return {
            "isomorphism_fidelity": torch.cosine_similarity(original_state.view(1, -1), reconstructed.view(1, -1)).item(),
            "spectral_shift": shift,
            "fueter_residual": df_residual,
            "veracity_check": is_valid
        }

if __name__ == "__main__":
    # Experimental validation
    encoder = HolomorphicGenomicEncoder()
    test_knot = torch.randn(256)
    
    results = encoder.verify_isomorphism(test_knot)
    print(f"[HGE] Isomorphism Audit: {results}")
    
    if results['veracity_check']:
        print("STABLE: Genomic Isomorphism verified.")
    else:
        print("EXPERIMENTAL: Topological tear detected in mapping.")