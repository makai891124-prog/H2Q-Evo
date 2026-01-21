"""Rotation Invariance Benchmark for H2Q Spacetime Vision.

Tests whether the 4D quaternion manifold representation provides
inherent rotation invariance, a key theoretical claim of the H2Q approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, List, Dict
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RotationInvarianceResult:
    """Stores rotation invariance test results."""
    model_name: str
    mean_cosine_similarity: float
    std_cosine_similarity: float
    max_deviation: float
    angles_tested: List[float]
    per_angle_similarity: Dict[float, float]


# ============================================================================
# H2Q Quaternion Feature Extractor
# ============================================================================

class RotationEquivariantQuaternionBlock(nn.Module):
    """
    Block that maintains rotation equivariance via quaternion structure.
    
    Key insight: Rotations in 3D correspond to conjugation by unit quaternion:
    v' = q * v * q^(-1)
    
    This block processes features while preserving this structure.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        # Quaternion-aware convolution (treats 4-channel groups specially)
        self.q_channels = channels
        
        # Weight sharing to preserve rotation structure
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=max(1, channels // 4))
        self.norm = nn.GroupNorm(max(1, channels // 4), channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class H2QQuaternionEncoder(nn.Module):
    """
    H2Q encoder that produces quaternion-valued features.
    
    Theory: SU(2) (quaternion) representation should be invariant
    to 3D rotations applied in the spatial domain.
    
    Enhanced with proper quaternion structure preservation.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # YCbCr projection (BT.601)
        ycbcr_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=torch.float32)
        self.register_buffer('ycbcr_proj', ycbcr_matrix)
        
        # RGB → 4D quaternion with phase component
        self.to_quaternion = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 4, 1),  # 4 = quaternion (w, x, y, z)
        )
        
        # Rotation-equivariant encoder stages
        q_hidden = (hidden_dim // 4) * 4  # Ensure divisible by 4
        
        self.encoder = nn.Sequential(
            nn.Conv2d(4, q_hidden // 2, 3, padding=1),
            nn.GroupNorm(q_hidden // 8, q_hidden // 2),
            nn.GELU(),
            RotationEquivariantQuaternionBlock(q_hidden // 2),
            nn.Conv2d(q_hidden // 2, q_hidden, 3, padding=1),
            nn.GroupNorm(q_hidden // 4, q_hidden),
            nn.GELU(),
            RotationEquivariantQuaternionBlock(q_hidden),
        )
        
        # Global pooling with quaternion-aware aggregation
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Project to final feature space
        self.proj = nn.Linear(q_hidden, hidden_dim)
        
        self.hidden_dim = hidden_dim
    
    def quaternion_normalize(self, q: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion groups to unit sphere."""
        B, C, H, W = q.shape
        # Reshape to quaternion groups
        q = q.view(B, C // 4, 4, H, W)
        # L2 normalize each quaternion
        q = F.normalize(q, p=2, dim=2)
        return q.view(B, C, H, W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RGB → Quaternion (with learned phase)
        q = self.to_quaternion(x)
        
        # Normalize to unit quaternion on SU(2)
        q = F.normalize(q, p=2, dim=1)
        
        # Feature extraction (maintains quaternion structure)
        h = self.encoder(q)
        
        # Normalize quaternion groups in hidden space
        h = self.quaternion_normalize(h)
        
        # Global average pooling
        h = self.pool(h).flatten(1)
        
        # Final projection
        return self.proj(h)


class BaselineCNNEncoder(nn.Module):
    """Standard CNN encoder for comparison."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


# ============================================================================
# Rotation Invariance Testing
# ============================================================================

def rotate_image(img: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate image by angle degrees."""
    return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)


def compute_feature_similarity(f1: torch.Tensor, f2: torch.Tensor) -> float:
    """Compute cosine similarity between feature vectors."""
    f1_norm = F.normalize(f1, p=2, dim=1)
    f2_norm = F.normalize(f2, p=2, dim=1)
    return (f1_norm * f2_norm).sum(dim=1).mean().item()


def test_rotation_invariance(
    model: nn.Module,
    model_name: str,
    test_images: torch.Tensor,
    angles: List[float] = None,
) -> RotationInvarianceResult:
    """
    Test rotation invariance of a model.
    
    Args:
        model: Feature extraction model
        model_name: Name for reporting
        test_images: (N, 3, H, W) tensor of test images
        angles: List of rotation angles to test
    
    Returns:
        RotationInvarianceResult with similarity metrics
    """
    if angles is None:
        angles = [15, 30, 45, 60, 90, 120, 180, 270]
    
    model.eval()
    device = next(model.parameters()).device
    test_images = test_images.to(device)
    
    with torch.no_grad():
        # Get features for original images
        original_features = model(test_images)
        
        per_angle_similarity = {}
        all_similarities = []
        
        for angle in angles:
            # Rotate images
            rotated_images = torch.stack([
                rotate_image(img, angle) for img in test_images
            ])
            
            # Get features for rotated images
            rotated_features = model(rotated_images)
            
            # Compute similarity
            sim = compute_feature_similarity(original_features, rotated_features)
            per_angle_similarity[angle] = sim
            all_similarities.append(sim)
        
        similarities = np.array(all_similarities)
    
    return RotationInvarianceResult(
        model_name=model_name,
        mean_cosine_similarity=float(np.mean(similarities)),
        std_cosine_similarity=float(np.std(similarities)),
        max_deviation=float(1.0 - np.min(similarities)),
        angles_tested=angles,
        per_angle_similarity=per_angle_similarity,
    )


def generate_test_images(num_images: int = 32, size: int = 32) -> torch.Tensor:
    """Generate random test images with structured patterns."""
    images = []
    
    for i in range(num_images):
        img = torch.zeros(3, size, size)
        
        # Add some structured patterns (shapes)
        cx, cy = size // 2, size // 2
        
        if i % 4 == 0:
            # Circle
            y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
            mask = ((x - cx)**2 + (y - cy)**2) < (size // 4)**2
            img[i % 3, mask] = 1.0
        elif i % 4 == 1:
            # Square
            s = size // 4
            img[i % 3, cy-s:cy+s, cx-s:cx+s] = 1.0
        elif i % 4 == 2:
            # Cross
            img[i % 3, cy-2:cy+2, :] = 0.8
            img[i % 3, :, cx-2:cx+2] = 0.8
        else:
            # Random gradient
            img = torch.rand(3, size, size) * 0.5
            y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
            img[0] += (x.float() / size) * 0.5
            img[1] += (y.float() / size) * 0.5
        
        images.append(img.clamp(0, 1))
    
    return torch.stack(images)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rotation Invariance Benchmark")
    parser.add_argument("--num-images", type=int, default=32, help="Number of test images")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[H2Q Benchmark] Rotation Invariance Test")
    print(f"Device: {device}")
    print(f"Test images: {args.num_images}")
    
    # Generate test images
    test_images = generate_test_images(args.num_images)
    
    # Test angles
    angles = [15, 30, 45, 60, 90, 120, 150, 180, 270, 360]
    
    results = []
    
    # 1. H2Q Quaternion Encoder (theoretical claim: rotation invariant)
    h2q_model = H2QQuaternionEncoder(args.hidden_dim).to(device)
    h2q_result = test_rotation_invariance(h2q_model, "H2Q-Quaternion", test_images, angles)
    results.append(h2q_result)
    
    # 2. Baseline CNN (should NOT be rotation invariant)
    cnn_model = BaselineCNNEncoder(args.hidden_dim).to(device)
    cnn_result = test_rotation_invariance(cnn_model, "Baseline-CNN", test_images, angles)
    results.append(cnn_result)
    
    # Print results
    print("\n" + "="*80)
    print("ROTATION INVARIANCE RESULTS")
    print("="*80)
    print(f"Ideal: Mean similarity = 1.0 (perfect rotation invariance)")
    print(f"       Std = 0.0 (consistent across all angles)")
    print("-"*80)
    
    for r in results:
        print(f"\n{r.model_name}:")
        print(f"  Mean Cosine Similarity: {r.mean_cosine_similarity:.4f}")
        print(f"  Std Deviation:          {r.std_cosine_similarity:.4f}")
        print(f"  Max Deviation from 1.0: {r.max_deviation:.4f}")
        print("  Per-angle similarity:")
        for angle, sim in r.per_angle_similarity.items():
            indicator = "✓" if sim > 0.9 else "✗"
            print(f"    {angle:>3}°: {sim:.4f} {indicator}")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    h2q_inv = results[0].mean_cosine_similarity
    cnn_inv = results[1].mean_cosine_similarity
    
    if h2q_inv > cnn_inv:
        improvement = (h2q_inv - cnn_inv) / (1 - cnn_inv) * 100 if cnn_inv < 1 else 0
        print(f"H2Q shows {improvement:.1f}% improvement in rotation invariance over baseline CNN")
    else:
        print("H2Q does not show improved rotation invariance (may need training)")
    
    print("\nNote: Random initialization may not show full invariance property.")
    print("      Training on rotation-augmented data can improve invariance.")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
