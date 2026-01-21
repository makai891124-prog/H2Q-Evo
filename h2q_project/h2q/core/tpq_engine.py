import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TPQMetrics:
    """Metrics for tracking quantization quality."""
    spectral_shift_mean: float = 0.0
    compression_ratio: float = 8.0  # 32-bit -> 8-bit default
    reconstruction_error: float = 0.0
    num_samples: int = 0


class DiscreteDecisionEngine(nn.Module):
    """
    STABLE: Fixed initialization to resolve 'dim' keyword error.
    The engine now accepts 'input_features' to define the manifold width.
    Supports batch operations and provides metrics tracking.
    """
    def __init__(self, input_features: int, num_phases: int = 256, device: Optional[str] = None):
        super().__init__()
        self.input_features = input_features
        self.num_phases = num_phases
        # Device selection with MPS/CUDA/CPU fallback
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        # Use a buffer for the phase-angle lookup table to ensure MPS compatibility
        phases = torch.linspace(0, 2 * np.pi, num_phases)
        self.register_buffer("phase_lut", phases)
        # Metrics tracking
        self._metrics = TPQMetrics()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discrete selection based on spectral shift η with batch support."""
        # Ensure input is on correct device
        x = x.to(self.phase_lut.device)
        return torch.bucketize(x, self.phase_lut)

    @property
    def metrics(self) -> TPQMetrics:
        return self._metrics

class TopologicalPhaseQuantizer(nn.Module):
    """
    EXPERIMENTAL: Implements SU(2) projection onto discrete phase-angles.
    Optimized for M4 Unified Memory (16GB) via 8:1 compression (L0 -> L1).
    
    Supports:
    - Batch quantization with automatic shape handling
    - Numerical stability guards for edge cases
    - Compression metrics tracking
    - Reversible encode/decode cycle
    """
    def __init__(self, bits: int = 8, device: Optional[str] = None):
        super().__init__()
        self.bits = bits
        self.levels = 2**bits
        # Symmetry Breaking parameter (h ± δ) - prevents acos domain errors
        self.delta = 1e-6
        # Device selection
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.decision_engine = DiscreteDecisionEngine(input_features=256, num_phases=self.levels, device=device)
        # Running metrics
        self._metrics = TPQMetrics(compression_ratio=32.0 / bits)
        self._sample_count = 0
        self._cumulative_eta = 0.0
        self._cumulative_error = 0.0

    def encode_su2_to_phase(self, q: torch.Tensor) -> torch.Tensor:
        """
        Projects a unit quaternion (SU(2)) into hyperspherical phase angles.
        q: [..., 4] (w, x, y, z)
        Returns: [..., 3] (psi, theta, phi) normalized to [0, 1]
        
        Handles batch dimensions automatically.
        """
        # Ensure unit norm for Geodesic Flow consistency
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        
        w, x, y, z = q.unbind(-1)
        
        # Hyperspherical mapping (Topological Spelling)
        # Numerical guard: clamp to avoid NaN from acos
        psi = torch.acos(w.clamp(-1 + self.delta, 1 - self.delta))
        sin_psi = torch.sin(psi).clamp(min=self.delta)
        
        # Additional guard for x / sin_psi
        ratio = (x / sin_psi).clamp(-1 + self.delta, 1 - self.delta)
        theta = torch.acos(ratio)
        phi = torch.atan2(z, y)
        
        # Normalize phases to [0, 1] for quantization
        psi_norm = psi / np.pi
        theta_norm = theta / np.pi
        phi_norm = (phi + np.pi) / (2 * np.pi)
        
        return torch.stack([psi_norm, theta_norm, phi_norm], dim=-1)

    def quantize(self, q: torch.Tensor, update_metrics: bool = True) -> torch.Tensor:
        """
        Rigid Construction: Maps continuous SU(2) to uint8 discrete manifold.
        
        Args:
            q: Input quaternions [..., 4]
            update_metrics: Whether to track compression quality
        
        Returns:
            Quantized phases as uint8 [..., 3]
        """
        phases = self.encode_su2_to_phase(q)
        # Scale to bit-depth and cast to uint8 for O(1) memory complexity
        quantized = (phases * (self.levels - 1)).round().clamp(0, self.levels - 1).to(torch.uint8)
        
        if update_metrics:
            # Compute reconstruction error for metrics
            with torch.no_grad():
                q_recon = self.dequantize(quantized)
                eta = self.get_spectral_shift(q, q_recon)
                error = torch.norm(q - q_recon, dim=-1).mean().item()
                batch_size = q.numel() // 4
                self._sample_count += batch_size
                self._cumulative_eta += eta.mean().item() * batch_size
                self._cumulative_error += error * batch_size
                self._update_metrics()
        
        return quantized

    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Elastic Weaving: Reconstructs the SU(2) manifold from discrete phases.
        
        Args:
            quantized: Quantized phases [..., 3] as uint8
        
        Returns:
            Reconstructed unit quaternions [..., 4]
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
        
        Returns:
            Spectral shift η in range [0, 1]
        """
        # Simplified scattering matrix trace for phase-only transitions
        inner_prod = torch.sum(q_orig * q_recon, dim=-1)
        eta = (1 / np.pi) * torch.acos(inner_prod.clamp(-1, 1))
        return eta

    def _update_metrics(self) -> None:
        """Update running metrics."""
        if self._sample_count > 0:
            self._metrics.spectral_shift_mean = self._cumulative_eta / self._sample_count
            self._metrics.reconstruction_error = self._cumulative_error / self._sample_count
            self._metrics.num_samples = self._sample_count

    @property
    def metrics(self) -> TPQMetrics:
        """Get current quantization metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._sample_count = 0
        self._cumulative_eta = 0.0
        self._cumulative_error = 0.0
        self._metrics = TPQMetrics(compression_ratio=32.0 / self.bits)

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encode-decode cycle for nn.Module compatibility.
        
        Returns:
            (quantized, reconstructed) tuple
        """
        quantized = self.quantize(q)
        reconstructed = self.dequantize(quantized)
        return quantized, reconstructed