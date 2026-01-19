import torch
from torch.optim import Optimizer
from h2q.quaternion_ops import quaternion_mul, quaternion_normalize
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class RAGO(Optimizer):
    """
    M4-Register-Aware Geodesic Optimizer (RAGO).
    Optimized for Mac Mini M4 AMX (16x16 tiling) and depth-12 fractal expansions.
    """
    def __init__(self, params, lr=1e-3, fractal_depth=12, tile_size=16, eta_target=0.1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(
            lr=lr, 
            fractal_depth=fractal_depth, 
            tile_size=tile_size,
            eta_target=eta_target
        )
        super(RAGO, self).__init__(params, defaults)
        
        # Initialize Metacognitive Components
        self.dde = get_canonical_dde() # Registry-safe initialization
        self.sst = SpectralShiftTracker()
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            depth = group['fractal_depth']
            tile_size = group['tile_size']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 1. Identify Atoms: Gradient in Tangent Space
                d_p = p.grad
                
                # 2. AMX-Tiled Processing to minimize register pressure
                # We process the manifold update in 16x16 blocks to fit M4 AMX cache lines
                self._apply_tiled_geodesic_update(p, d_p, lr, depth, tile_size)

        return loss

    def _apply_tiled_geodesic_update(self, p, d_p, lr, depth, tile_size):
        """
        Performs Geodesic Flow update using SU(2) exponential map with AMX tiling hints.
        """
        shape = p.shape
        # Ensure we are working with quaternionic representations (last dim 4)
        if shape[-1] != 4:
            # Fallback for non-quaternionic parameters
            p.add_(d_p, alpha=-lr)
            return

        # Flatten to process as tiles
        flat_p = p.view(-1, 4)
        flat_grad = d_p.view(-1, 4)
        num_elements = flat_p.size(0)

        for i in range(0, num_elements, tile_size):
            end_idx = min(i + tile_size, num_elements)
            
            # Extract Tile (16x16 hint: 16 quaternions = 64 floats, fits AMX registers)
            p_tile = flat_p[i:end_idx]
            g_tile = flat_grad[i:end_idx]

            # 3. Fractal Expansion Protocol (h ± δ) over 12 levels
            # We compute the geodesic displacement delta
            delta = g_tile * lr
            
            for d in range(1, depth + 1):
                # Scale delta fractally: delta_d = delta / (2^d)
                scale = 1.0 / (2.0 ** d)
                
                # Geodesic Flow: exp_map(v) = [cos(|v|), (v/|v|) * sin(|v|)]
                v_norm = torch.norm(delta * scale, dim=-1, keepdim=True) + 1e-8
                exp_w = torch.cos(v_norm)
                exp_xyz = (delta * scale / v_norm) * torch.sin(v_norm)
                exp_q = torch.cat([exp_w, exp_xyz], dim=-1)

                # 4. Hamilton Product (AMX Optimized via tiling)
                # p_new = p_old * exp_q
                p_tile.copy_(quaternion_mul(p_tile, exp_q))

            # 5. Symmetry Verification: Maintain S³ Manifold
            p_tile.copy_(quaternion_normalize(p_tile))

    def get_spectral_health(self):
        """
        Returns the current η (Spectral Shift) to monitor manifold stability.
        """
        return self.sst.get_eta()