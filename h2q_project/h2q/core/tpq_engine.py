import torch
import torch.nn as nn
import numpy as np

class DiscreteDecisionEngine(nn.Module):
    """
    STABLE: Fixed initialization to resolve 'dim' keyword error.
    The engine now accepts 'input_features' to define the manifold width.
    """
    def __init__(self, input_features: int, num_phases: int = 256):
        super().__init__()
        self.input_features = input_features
        self.num_phases = num_phases
        # Use a buffer for the phase-angle lookup table to ensure MPS compatibility
        phases = torch.linspace(0, 2 * np.pi, num_phases)
        self.register_buffer("phase_lut", phases)

    def forward(self, x):
        # Logic for discrete selection based on spectral shift η
        return torch.bucketize(x, self.phase_lut)

class TopologicalPhaseQuantizer(nn.Module):
    """
    EXPERIMENTAL: Implements SU(2) projection onto discrete phase-angles.
    Optimized for M4 Unified Memory (16GB) via 8:1 compression (L0 -> L1).
    """
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.levels = 2**bits
        # Symmetry Breaking parameter (h ± δ)
        self.delta = 1e-6 
        self.decision_engine = DiscreteDecisionEngine(input_features=256, num_phases=self.levels)

    def encode_su2_to_phase(self, q: torch.Tensor) -> torch.Tensor:
        """
        Projects a unit quaternion (SU(2)) into hyperspherical phase angles.
        q: [..., 4] (w, x, y, z)
        Returns: [..., 3] (psi, theta, phi) normalized to [0, 1]
        """
        # Ensure unit norm for Geodesic Flow consistency
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        
        w, x, y, z = q.unbind(-1)
        
        # Hyperspherical mapping (Topological Spelling)
        psi = torch.acos(w.clamp(-1 + self.delta, 1 - self.delta))
        sin_psi = torch.sin(psi).clamp(min=self.delta)
        
        theta = torch.acos((x / sin_psi).clamp(-1 + self.delta, 1 - self.delta))
        phi = torch.atan2(z, y)
        
        # Normalize phases to [0, 1] for quantization
        psi_norm = psi / np.pi
        theta_norm = theta / np.pi
        phi_norm = (phi + np.pi) / (2 * np.pi)
        
        return torch.stack([psi_norm, theta_norm, phi_norm], dim=-1)

    def quantize(self, q: torch.Tensor) -> torch.Tensor:
        """
        Rigid Construction: Maps continuous SU(2) to uint8 discrete manifold.
        """
        phases = self.encode_su2_to_phase(q)
        # Scale to bit-depth and cast to uint8 for O(1) memory complexity
        quantized = (phases * (self.levels - 1)).to(torch.uint8)
        return quantized

    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Elastic Weaving: Reconstructs the SU(2) manifold from discrete phases.
        """
        phases = quantized.to(torch.float32) / (self.levels - 1)
        
        psi = phases[..., 0] * np.pi
        theta = phases[..., 1] * np.pi
        phi = (phases[..., 2] * 2 * np.pi) - np.pi
        
        # Reconstruct Quaternion components
        w = torch.cos(psi)
        sin_psi = torch.sin(psi)
        x = sin_psi * torch.cos(theta)
        y = sin_psi * torch.sin(theta) * torch.cos(phi)
        z = sin_psi * torch.sin(theta) * torch.sin(phi)
        
        return torch.stack([w, x, y, z], dim=-1)

    def get_spectral_shift(self, q_orig: torch.Tensor, q_recon: torch.Tensor) -> torch.Tensor:
        """
        Calculates η = (1/π) arg{det(S)} via the Krein-like trace formula.
        Used to track information loss during quantization.
        """
        # Simplified scattering matrix trace for phase-only transitions
        inner_prod = torch.sum(q_orig * q_recon, dim=-1)
        eta = (1/np.pi) * torch.acos(inner_prod.clamp(-1, 1))
        return eta