import asyncio
import json
import torch
from typing import List, Dict, Any
from h2q.monitoring.mhdm import ManifoldHeatDeathMonitor
from h2q.visualization.fueter_poincare_dashboard import PoincareFueterDashboard
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class HolomorphicTelemetryMiddleware:
    """
    Middleware for h2q_server to provide real-time WebSocket streams of 
    Poincare Curvature and Heat-Death Index (HDI).
    """
    def __init__(self):
        # Use registry to avoid 'dim' keyword error in DiscreteDecisionEngine
        self.dde = get_canonical_dde()
        self.mhdm = ManifoldHeatDeathMonitor(self.dde)
        self.dashboard = PoincareFueterDashboard()
        self.sst = SpectralShiftTracker()
        self.active_connections: List[Any] = []

    async def connect(self, websocket: Any):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: Any):
        self.active_connections.remove(websocket)

    def _calculate_hdi(self, manifold_state: torch.Tensor) -> float:
        """
        Calculates Heat-Death Index (HDI) based on Spectral Shift (η).
        η = (1/π) arg{det(S)}
        """
        # η is the spectral shift tracker output
        eta = self.sst.calculate_spectral_shift(manifold_state)
        # HDI is normalized environmental drag integration
        hdi = torch.tanh(eta).item()
        return hdi

    def _calculate_poincare_curvature(self, manifold_state: torch.Tensor) -> Dict[str, Any]:
        """
        Computes the Poincare Curvature Map using the Discrete Fueter Operator.
        Df = ∂w + i∂x + j∂y + k∂z
        """
        # Identify topological tears where Df > 0.05
        # This logic is delegated to the dashboard for visualization prep
        curvature_map = self.dashboard.generate_curvature_data(manifold_state)
        return curvature_map

    async def broadcast_telemetry(self, manifold_state: torch.Tensor):
        """
        Broadcasts real-time metrics to all connected WebSockets.
        """
        if not self.active_connections:
            return

        hdi = self._calculate_hdi(manifold_state)
        curvature = self._calculate_poincare_curvature(manifold_state)

        payload = {
            "type": "holomorphic_telemetry",
            "metrics": {
                "heat_death_index": hdi,
                "is_stable": hdi < 0.8,
                "spectral_shift": hdi * 3.14159 # Approximation of η
            },
            "visuals": {
                "poincare_map": curvature
            }
        }

        message = json.dumps(payload)
        
        # Broadcast to all active listeners
        tasks = [connection.send_text(message) for connection in self.active_connections]
        if tasks:
            await asyncio.gather(*tasks)

    async def tap_inference(self, manifold_generator):
        """
        Async generator wrapper to tap into the inference flow.
        """
        async for state in manifold_generator:
            # Non-blocking broadcast
            asyncio.create_task(self.broadcast_telemetry(state))
            yield state

# Stable implementation for h2q_server integration
def get_telemetry_middleware():
    return HolomorphicTelemetryMiddleware()