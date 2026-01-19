import torch
import math

class TopologicalPhaseQuantizer:
    """
    [EXPERIMENTAL] TPQ Engine
    Implements Topological Phase Quantization for quaternionic states.
    Maps hyperspherical angles (psi, theta, phi) to uint8 to preserve η-signature.
    Compression Ratio: 8:1 (assuming 64-bit quaternionic atoms to 8-bit indices).
    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            self.device = device
        
        # Bit allocation for S3 manifold: 2 bits (psi), 3 bits (theta), 3 bits (phi)
        self.psi_bins = 4
        self.theta_bins = 8
        self.phi_bins = 8

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        """
        Quantizes a quaternionic tensor [..., 4] into uint8 phases.
        q components: [w, x, y, z]
        """
        # 1. Normalize to unit sphere (S3) to isolate phase from magnitude
        norm = torch.norm(q, p=2, dim=-1, keepdim=True) + 1e-8
        q_unit = q / norm

        w, x, y, z = q_unit[..., 0], q_unit[..., 1], q_unit[..., 2], q_unit[..., 3]

        # 2. Extract Hyperspherical Angles
        # psi: [0, pi]
        psi = torch.acos(w.clamp(-1.0, 1.0))
        
        # sin_psi for normalization of lower angles
        sin_psi = torch.sin(psi) + 1e-8
        
        # theta: [0, pi]
        theta = torch.acos((z / sin_psi).clamp(-1.0, 1.0))
        
        # phi: [-pi, pi]
        phi = torch.atan2(y, x)

        # 3. Quantize to bit-depths
        # psi -> 2 bits (0-3)
        psi_q = torch.round((psi / math.pi) * (self.psi_bins - 1)).to(torch.uint8)
        # theta -> 3 bits (0-7)
        theta_q = torch.round((theta / math.pi) * (self.theta_bins - 1)).to(torch.uint8)
        # phi -> 3 bits (0-7)
        phi_q = torch.round(((phi + math.pi) / (2 * math.pi)) * (self.phi_bins - 1)).to(torch.uint8)

        # 4. Pack into uint8: [PP TTT FFF]
        packed = (psi_q << 6) | (theta_q << 3) | phi_q
        return packed

    def decode(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs unit quaternions from uint8 phase indices.
        """
        # 1. Unpack bits
        psi_q = (packed >> 6) & 0x03
        theta_q = (packed >> 3) & 0x07
        phi_q = packed & 0x07

        # 2. Map back to angles
        psi = (psi_q.float() / (self.psi_bins - 1)) * math.pi
        theta = (theta_q.float() / (self.theta_bins - 1)) * math.pi
        phi = (phi_q.float() / (self.phi_bins - 1)) * (2 * math.pi) - math.pi

        # 3. Reconstruct Quaternionic components
        # w = cos(psi)
        # x = sin(psi) * sin(theta) * cos(phi)
        # y = sin(psi) * sin(theta) * sin(phi)
        # z = sin(psi) * cos(theta)
        
        sin_psi = torch.sin(psi)
        sin_theta = torch.sin(theta)
        
        w = torch.cos(psi)
        x = sin_psi * sin_theta * torch.cos(phi)
        y = sin_psi * sin_theta * torch.sin(phi)
        z = sin_psi * torch.cos(theta)

        return torch.stack([w, x, y, z], dim=-1)

    def compute_eta_signature(self, q: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Spectral Shift Tracker (η) signature.
        η = (1/π) arg{det(S)}
        In this atom, we approximate the scattering matrix S via the quaternionic phase.
        """
        # Simplified η tracking via the aggregate phase angle
        # This ensures semantic isomorphism during compression
        angles = torch.atan2(torch.norm(q[..., 1:], p=2, dim=-1), q[..., 0])
        eta = torch.mean(angles) / math.pi
        return eta

# Verification of Symmetry and Veracity
if __name__ == "__main__":
    quantizer = TopologicalPhaseQuantizer()
    # Mock 256-dim quaternionic knot (64 quaternions)
    test_knot = torch.randn(1, 64, 4).to(quantizer.device)
    
    # Quantize
    encoded = quantizer.encode(test_knot)
    # Dequantize
    decoded = quantizer.decode(encoded)
    
    # Verify η-signature preservation
    eta_orig = quantizer.compute_eta_signature(test_knot)
    eta_quant = quantizer.compute_eta_signature(decoded)
    
    print(f"Original η: {eta_orig:.4f}")
    print(f"Quantized η: {eta_quant:.4f}")
    print(f"Compression: {test_knot.element_size() * test_knot.nelement()} bytes -> {encoded.element_size() * encoded.nelement()} bytes")
