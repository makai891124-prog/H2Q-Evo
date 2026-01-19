import torch
import psutil
import os
from typing import Optional
from h2q.knot_kernel import H2Q_Knot_Kernel
from h2q.core.interface_registry import get_canonical_dde

class DynamicManifoldDepthController(torch.nn.Module):
    """
    DMDC: Dynamic Manifold Depth Controller.
    Wraps H2Q_Knot_Kernel to dynamically prune recursion depth based on RSS telemetry.
    Optimized for Mac Mini M4 (16GB) constraints.
    """
    def __init__(
        self,
        knot_kernel: H2Q_Knot_Kernel,
        memory_threshold_gb: float = 12.0,
        critical_threshold_gb: float = 14.5,
        max_depth: int = 12,
        min_depth: int = 4
    ):
        super().__init__()
        self.kernel = knot_kernel
        self.memory_threshold = memory_threshold_gb * 1024**3
        self.critical_threshold = critical_threshold_gb * 1024**3
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        # Initialize DDE without 'dim' to avoid registry-reported init errors
        self.dde = get_canonical_dde()
        self.process = psutil.Process(os.getpid())

    def get_rss_telemetry(self) -> float:
        """Returns current Resident Set Size in bytes."""
        return float(self.process.memory_info().rss)

    def compute_dynamic_depth(self) -> int:
        """
        Maps RSS pressure to recursion depth [4, 12].
        Linear pruning between threshold and critical limit.
        """
        current_rss = self.get_rss_telemetry()
        
        if current_rss < self.memory_threshold:
            return self.max_depth
        
        if current_rss >= self.critical_threshold:
            return self.min_depth

        # Linear interpolation of depth pruning
        pressure_ratio = (current_rss - self.memory_threshold) / (self.critical_threshold - self.memory_threshold)
        depth = int(self.max_depth - (pressure_ratio * (self.max_depth - self.min_depth)))
        return max(self.min_depth, depth)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Executes the Knot_Kernel with dynamic depth pruning.
        """
        target_depth = self.compute_dynamic_depth()
        
        # Logic Veracity Check: Ensure we don't cause a topological tear
        # by pruning too aggressively without notifying the decision engine.
        decision = self.dde.decide(state=x, options=torch.tensor([target_depth]))
        
        # We assume H2Q_Knot_Kernel forward accepts a 'depth' or 'iterations' parameter.
        # If not, we manually iterate the kernel's internal logic atoms.
        out = x
        for _ in range(target_depth):
            out = self.kernel(out, *args, **kwargs)
            
        return out

def stream_with_dmdc(data_stream, kernel: H2Q_Knot_Kernel, controller: Optional[DynamicManifoldDepthController] = None):
    """
    Utility function to process a stream with DMDC protection.
    """
    if controller is None:
        controller = DynamicManifoldDepthController(kernel)
        
    for batch in data_stream:
        yield controller(batch)