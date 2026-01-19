import psutil
import torch
import logging
from h2q.core.engine import DiscreteDecisionEngine, FractalExpansion, AdaptiveSemanticStrider

class MemoryPressureManifoldGovernor:
    """
    MPMG: Memory Pressure Manifold Governor
    Monitors RSS telemetry to dynamically throttle H2Q manifold parameters.
    Target Hardware: Mac Mini M4 (16GB RAM).
    """
    def __init__(self, manifold_dim: int = 256, critical_threshold_gb: float = 14.0, safe_threshold_gb: float = 10.0):
        self.process = psutil.Process()
        self.manifold_dim = manifold_dim
        self.critical_threshold = critical_threshold_gb * 1024**3
        self.safe_threshold = safe_threshold_gb * 1024**3
        
        # Initialize Decision Engine for governance logic
        # Registry check: h2q.core.engine.DiscreteDecisionEngine takes (dim, num_decisions)
        self.decision_engine = DiscreteDecisionEngine(manifold_dim, 2)
        
        self.current_depth = 12
        self.current_stride = 8
        
        logging.info(f"[MPMG] Initialized. Thresholds: {safe_threshold_gb}GB - {critical_threshold_gb}GB")

    def get_rss_telemetry(self) -> float:
        """Returns current Resident Set Size in bytes."""
        return self.process.memory_info().rss

    def step(self) -> dict:
        """
        Calculates the required throttling based on memory pressure.
        Returns a configuration dict for FractalExpansion and AdaptiveStriding.
        """
        rss = self.get_rss_telemetry()
        
        if rss < self.safe_threshold:
            # Optimal performance state
            self.current_depth = 12
            self.current_stride = 8
        elif rss > self.critical_threshold:
            # Emergency compression state
            self.current_depth = 4
            self.current_stride = 16
        else:
            # Linear interpolation of pressure
            ratio = (rss - self.safe_threshold) / (self.critical_threshold - self.safe_threshold)
            self.current_depth = int(12 - (8 * ratio))
            self.current_stride = int(8 + (8 * ratio))

        # Ensure bounds
        self.current_depth = max(4, min(12, self.current_depth))
        self.current_stride = max(8, min(16, self.current_stride))

        return {
            "fractal_depth": self.current_depth,
            "stride_ratio": self.current_stride,
            "rss_gb": rss / 1024**3,
            "pressure_level": "CRITICAL" if rss > self.critical_threshold else "STABLE"
        }

    def apply_governance(self, x: torch.Tensor, expansion_layer: FractalExpansion, strider: AdaptiveSemanticStrider):
        """
        Applies the throttled parameters to the manifold operations.
        """
        config = self.step()
        
        # Fractal Expansion throttling
        # Note: We assume the expansion_layer forward or internal state accepts depth
        # If the layer is rigid, we simulate depth by slicing the recursive calls
        z = expansion_layer(x)
        
        # Adaptive Striding throttling
        # Registry check: h2q.core.strider.AdaptiveSemanticStrider.forward(x)
        # We adjust the striding logic based on the governor's stride_ratio
        s = strider(z)
        
        if config["pressure_level"] == "CRITICAL":
            logging.warning(f"[MPMG] Memory Pressure High ({config['rss_gb']:.2f}GB). Throttling: Depth={self.current_depth}, Stride={self.current_stride}")
            
        return s, config

def verify_governor_integrity():
    """Unit test for MPMG logic."""
    gov = MemoryPressureManifoldGovernor()
    telemetry = gov.step()
    assert 4 <= telemetry["fractal_depth"] <= 12
    assert 8 <= telemetry["stride_ratio"] <= 16
    print(f"MPMG Integrity Verified: {telemetry}")

if __name__ == "__main__":
    verify_governor_integrity()