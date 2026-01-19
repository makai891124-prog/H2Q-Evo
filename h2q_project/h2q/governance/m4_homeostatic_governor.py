import torch
import time
from typing import Dict, Any, Tuple
from h2q.core.interface_registry import get_canonical_dde, normalize_dde_kwargs
from h2q.core.ttd_scheduler import TopologicalTimeDilation, TTDState
from h2q.dispatch.amx_tiling_dispatcher import DynamicAMXTilingDispatcher
from h2q.core.monitoring.manifold_audit import ManifoldAuditor

class M4HomeostaticGovernor:
    """
    Unified controller for Mac Mini M4 (MPS/16GB).
    Monitors thermal pressure and SLC cache telemetry to adjust TTD recursion and AMX tiling.
    """
    def __init__(self, 
                 ttd_scheduler: TopologicalTimeDilation,
                 amx_dispatcher: DynamicAMXTilingDispatcher,
                 target_temp_threshold: float = 0.75):
        self.ttd = ttd_scheduler
        self.amx = amx_dispatcher
        self.auditor = ManifoldAuditor()
        self.target_temp = target_temp_threshold
        
        # Initialize DDE using the canonical registry to avoid 'dim' keyword errors
        dde_kwargs = normalize_dde_kwargs(alpha=0.8) 
        self.dde = get_canonical_dde(**dde_kwargs)
        
        self.history = {"thermal": [], "slc_pressure": [], "ttd_depth": []}

    def poll_telemetry(self) -> Dict[str, float]:
        """
        Simulates/Retrieves M4 hardware telemetry.
        In a production MPS environment, this would interface with IOKit or sysctl.
        """
        # Mocking telemetry based on current manifold curvature (proxy for compute load)
        curvature = self.auditor.measure_logic_curvature() if hasattr(self.auditor, 'measure_logic_curvature') else 0.5
        
        # SLC Pressure: High curvature/long context increases cache misses
        slc_pressure = torch.rand(1).item() * curvature 
        
        # Thermal Pressure: Simulated thermal ramp
        thermal_pressure = torch.sigmoid(torch.tensor([slc_pressure * 2.0])).item()
        
        return {
            "thermal": thermal_pressure,
            "slc": slc_pressure
        }

    def compute_homeostatic_shift(self, telemetry: Dict[str, float]) -> Tuple[int, int]:
        """
        Uses Discrete Decision Engine to select optimal (TTD_Depth, AMX_Tile_Size).
        """
        thermal = telemetry["thermal"]
        slc = telemetry["slc"]
        
        # Decision Space: 
        # 0: High Performance (Depth 8, Tile 32)
        # 1: Balanced (Depth 4, Tile 16)
        # 2: Thermal Recovery (Depth 1, Tile 8)
        
        # We pass the state to DDE. Note: DDE expects a loss/utility context.
        # Utility = (1 - thermal) * Veracity_Weight
        decision_idx = self.dde.decide(torch.tensor([thermal, slc]))
        
        if thermal > self.target_temp or decision_idx == 2:
            return 1, 8   # Recovery Mode
        elif thermal > 0.5 or decision_idx == 1:
            return 4, 16  # Balanced Mode
        else:
            return 8, 32  # Max Veracity Mode

    def step(self):
        """
        Execution cycle of the governor.
        """
        telemetry = self.poll_telemetry()
        new_depth, new_tile = self.compute_homeostatic_shift(telemetry)
        
        # Apply TTD Depth Adjustment
        if hasattr(self.ttd, 'state'):
            self.ttd.state.recursion_depth = new_depth
        
        # Apply AMX Tiling Adjustment
        if hasattr(self.amx, 'update_tile_config'):
            self.amx.update_tile_config(tile_size=new_tile)
            
        # Log for persistence audit
        self.history["thermal"].append(telemetry["thermal"])
        self.history["slc_pressure"].append(telemetry["slc"])
        self.history["ttd_depth"].append(new_depth)

    def get_status(self) -> str:
        return f"[M4-GOVERNOR] Thermal: {self.history['thermal'][-1]:.2f} | TTD-Depth: {self.history['ttd_depth'][-1]} | AMX-Tile: {self.amx.current_tile if hasattr(self.amx, 'current_tile') else 'N/A'}"

def initialize_m4_governor(ttd, amx):
    return M4HomeostaticGovernor(ttd, amx)