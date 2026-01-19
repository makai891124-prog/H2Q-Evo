import torch
import numpy as np
from typing import Tuple, Dict
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class M4RegisterTelemetry:
    """
    Monitors NPU/GPU register pressure and memory bandwidth on Apple Silicon M4.
    In a production environment, this interfaces with Metal Performance Shaders (MPS) 
    counter sets. Here, it tracks simulated pressure based on tensor occupancy.
    """
    def __init__(self):
        self.base_pressure = 0.1
        self.peak_registers = 256 # M4 typical SIMD register count per thread group

    def get_current_pressure(self, tensor_shape: torch.Size) -> float:
        # Calculate occupancy: larger tensors increase register pressure
        occupancy = (tensor_shape.numel() * 4) / (16 * 1024 * 1024 * 1024) # Normalized to 16GB
        pressure = min(1.0, self.base_pressure + occupancy * 10)
        return pressure

class DynamicAMXTilingDispatcher:
    """
    M4-Register-Aware Tiling Scheduler.
    Dynamically adjusts Metal Shader tiling sizes (8x8 to 32x32) based on 
    real-time NPU register pressure and manifold entropy (HDI).
    """
    def __init__(self, alpha: float = 0.5):
        # Initialize the Canonical Discrete Decision Engine
        # Fixed the 'dim' error by using the registry-standardized wrapper
        self.dde = get_canonical_dde(n_actions=3) # Actions: 0: 8x8, 1: 16x16, 2: 32x32
        self.telemetry = M4RegisterTelemetry()
        self.sst = SpectralShiftTracker()
        self.alpha = alpha # Weight for register pressure vs entropy
        
        self.tile_map = {
            0: (8, 8),
            1: (16, 16),
            2: (32, 32)
        }

    def compute_system_stress(self, register_pressure: float, manifold_entropy: float) -> torch.Tensor:
        """
        Calculates the combined stress index (Heat-Death Index approximation).
        """
        # High entropy or high pressure increases stress
        stress = self.alpha * register_pressure + (1 - self.alpha) * manifold_entropy
        return torch.tensor([stress], dtype=torch.float32)

    def get_optimal_tiling(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        """
        Determines the optimal tile size for the current manifold state.
        """
        # 1. Gather Telemetry
        reg_pressure = self.telemetry.get_current_pressure(input_tensor.shape)
        
        # 2. Gather Manifold Entropy (Spectral Shift η)
        # We use the tracker to see if the logic curvature is deviating
        manifold_entropy = self.sst.get_eta() 
        
        # 3. Construct State for DDE
        state = self.compute_system_stress(reg_pressure, manifold_entropy)
        
        # 4. Decision Logic:
        # If stress is high (> 0.7), DDE should bias towards smaller tiles (8x8) 
        # to prevent register spills and maintain O(1) memory complexity.
        # If stress is low, 32x32 maximizes AMX throughput.
        action = self.dde.decide(state)
        
        # 5. Safety Override: If pressure is critical, force minimum tiling
        if reg_pressure > 0.9:
            return self.tile_map[0]
            
        return self.tile_map[action.item()]

    def dispatch_kernel(self, kernel_fn, input_tensor: torch.Tensor, *args, **kwargs):
        """
        Executes a kernel with the dynamically selected tiling strategy.
        """
        tile_size = self.get_optimal_tiling(input_tensor)
        
        # Log the dispatch for the Veracity Compact audit
        # print(f"[M4-AMX] Dispatching with TileSize: {tile_size} | Pressure: {self.telemetry.get_current_pressure(input_tensor.shape):.2f}")
        
        return kernel_fn(input_tensor, tile_size=tile_size, *args, **kwargs)

# Experimental: Symmetry Validation
def verify_tiling_symmetry(scheduler: DynamicAMXTilingDispatcher):
    """
    Ensures that the scheduler responds inversely to pressure.
    """
    low_pressure_tensor = torch.randn(1, 256) # Small
    high_pressure_tensor = torch.randn(1024, 1024, 64) # Large
    
    tile_low = scheduler.get_optimal_tiling(low_pressure_tensor)
    tile_high = scheduler.get_optimal_tiling(high_pressure_tensor)
    
    # Symmetry check: High pressure should never result in larger tiles than low pressure
    assert tile_high[0] <= tile_low[0], "Tiling Symmetry Broken: High pressure assigned large tiles."
    return True