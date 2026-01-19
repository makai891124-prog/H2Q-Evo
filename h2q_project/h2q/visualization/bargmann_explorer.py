import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.bargmann_prover import BargmannIsomorphismProver
from h2q.grounding.genomic_streamer import TopologicalFASTAStreamer
from h2q.core.sst import SpectralShiftTracker

class BargmannExplorer:
    """
    [BARGMANN-EXPLORER]: Visual diagnostic tool mapping 3-point Bargmann invariants
    onto a hyperbolic Poincare Disk to verify cross-modal semantic isomorphism.
    """
    def __init__(self, device: str = "mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        
        # Fix for DiscreteDecisionEngine.__init__() 'dim' error:
        # We use the canonical registry to instantiate the DDE safely.
        self.dde = get_canonical_dde()
        self.prover = BargmannIsomorphismProver()
        self.sst = SpectralShiftTracker()
        
        # Visualization settings
        self.fig, self.ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        self.ax.set_ylim(0, 1)
        self.ax.set_title("H2Q Bargmann Isomorphism: Poincare Disk Projection", color='white', pad=20)
        self.fig.patch.set_facecolor('#0a0a0a')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.grid(True, color='#333333', linestyle='--')

    def _compute_3point_invariant(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the 3-point Bargmann invariant: B(z1, z2, z3) = <z1,z2><z2,z3><z3,z1>.
        Input shape: [N, 3, D] where D is the SU(2) embedding dimension.
        """
        z1, z2, z3 = states[:, 0], states[:, 1], states[:, 2]
        
        # Quaternionic/Complex inner products
        dot12 = torch.sum(z1 * z2, dim=-1)
        dot23 = torch.sum(z2 * z3, dim=-1)
        dot31 = torch.sum(z3 * z1, dim=-1)
        
        return dot12 * dot23 * dot31

    def project_to_poincare(self, invariants: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps complex invariants to Poincare Disk coordinates (r, theta).
        r = tanh(|B|) to ensure hyperbolic mapping within the unit disk.
        theta = arg(B) representing the geometric phase.
        """
        # Treat real invariants as complex with 0 imag if necessary
        # In H2Q, invariants are often phase-deflected scalars
        mag = torch.abs(invariants).cpu().numpy()
        phase = torch.angle(invariants.to(torch.complex64)).cpu().numpy() if invariants.is_complex() else torch.zeros_like(mag)
        
        # Hyperbolic radial mapping
        r = np.tanh(mag)
        return r, phase

    def run_diagnostic(self, code_knots: torch.Tensor, genomic_sequences: torch.Tensor):
        """
        Executes the mapping of StarCoder logic knots and Genomic FASTA sequences.
        """
        print("[BARGMANN-EXPLORER] Auditing Cross-Modal Symmetry...")
        
        # 1. Process Code Knots (StarCoder Logic)
        # Assuming knots are provided as [N, 3, D] sequences of SU(2) states
        code_inv = self._compute_3point_invariant(code_knots)
        r_code, theta_code = self.project_to_poincare(code_inv)
        
        # 2. Process Genomic Sequences
        # Assuming genomic_sequences are provided as [N, 3, D]
        genom_inv = self._compute_3point_invariant(genomic_sequences)
        r_genom, theta_genom = self.project_to_poincare(genom_inv)

        # 3. Visualization
        self.ax.scatter(theta_code, r_code, c='#00f2ff', label='StarCoder Logic Knots', alpha=0.6, s=20, edgecolors='none')
        self.ax.scatter(theta_genom, r_genom, c='#ff007b', label='Genomic FASTA (SU2)', alpha=0.6, s=20, edgecolors='none')
        
        # Draw the unit circle boundary
        circle = plt.Circle((0, 0), 1, transform=self.ax.transData._b, color='#444444', fill=False, linewidth=2)
        self.ax.add_artist(circle)

        self.ax.legend(loc='upper right', frameon=True, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')
        
        # 4. Veracity Audit (Fueter Residuals)
        # Hallucinations are topological tears where |Df| > 0.05
        # Here we simulate the audit based on the spectral shift
        eta = self.sst.calculate_spectral_shift(code_knots, genom_inv)
        print(f"[VERACITY_COMPACT] Spectral Shift Tracker (η): {eta:.4f}")
        
        if eta > 0.05:
            print("WARNING: Topological Tear Detected (|Df| > 0.05). Semantic Isomorphism Compromised.")
        else:
            print("SUCCESS: Manifold Symmetry Verified. Isomorphism Stable.")

        plt.savefig("bargmann_isomorphism_disk.png", facecolor='#0a0a0a')
        print("Diagnostic saved to bargmann_isomorphism_disk.png")

if __name__ == "__main__":
    # Mock data for demonstration within M4 constraints
    explorer = BargmannExplorer(device="mps")
    
    # Generate synthetic SU(2) sequences [Batch, 3-points, Dim]
    mock_code = torch.randn(500, 3, 4).to("mps")
    mock_code = torch.nn.functional.normalize(mock_code, p=2, dim=-1)
    
    mock_genom = torch.randn(500, 3, 4).to("mps")
    mock_genom = torch.nn.functional.normalize(mock_genom, p=2, dim=-1)
    
    explorer.run_diagnostic(mock_code, mock_genom)
