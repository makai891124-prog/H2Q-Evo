import torch
import torch.nn as nn
from h2q.core.interface_registry import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

class UnitaryRecoveryHook:
    """
    [STABLE] Unitary Recovery Hook
    Applies periodic QR-reorthogonalization to linear layer weights to neutralize 
    floating-point drift and enforce SU(2) / Unitary manifold constraints.
    """
    def __init__(self, model: nn.Module, recovery_interval: int = 100, eta_threshold: float = 0.05):
        self.model = model
        self.recovery_interval = recovery_interval
        self.eta_threshold = eta_threshold
        self.step_count = 0
        
        # Initialize DDE using canonical registry to avoid 'dim' keyword errors
        # The registry handles the normalization of arguments for the specific DDE version.
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()

    @torch.no_grad()
    def __call__(self, module, input):
        """
        Hook signature for register_forward_pre_hook.
        """
        self.step_count += 1
        if self.step_count % self.recovery_interval == 0:
            self.recover_manifold()

    @torch.no_grad()
    def recover_manifold(self):
        """
        Performs QR-based projection of weights back onto the Stiefel/Unitary manifold.
        For H2Q, this ensures the 256-dimensional topological manifold remains rigid.
        """
        for name, param in self.model.named_parameters():
            # Only target weights of linear or quaternionic layers
            if 'weight' in name and param.ndim >= 2:
                # 1. Identify Atom: Extract weight matrix
                w = param.data
                original_shape = w.shape
                
                # Flatten to 2D if it's a conv kernel or higher-dim tensor
                if w.ndim > 2:
                    w = w.view(w.size(0), -1)

                # 2. Verify Symmetry: QR decomposition
                # We use 'reduced' mode to handle non-square matrices (isometries)
                # On M4 (MPS), torch.linalg.qr is optimized for the AMX unit.
                try:
                    q, r = torch.linalg.qr(w, mode='reduced')
                    
                    # 3. SU(2) Constraint Enforcement:
                    # In H2Q, we treat the manifold as a collection of SU(2) knots.
                    # For a general weight matrix, Q provides the closest orthonormal matrix.
                    param.data.copy_(q.view(original_shape))
                    
                    # 4. Spectral Shift Audit:
                    # Calculate the 'jump' in the manifold to update the tracker η.
                    # η = (1/π) arg{det(S)}
                    shift = torch.norm(w - q.view(original_shape)) / torch.norm(w)
                    if shift > self.eta_threshold:
                        self.sst.update_eta(shift.item())
                        
                except RuntimeError as e:
                    # Embrace Noise: Log failure but do not crash; the manifold may be singular.
                    print(f"[M24-CW] Manifold Recovery Warning on {name}: {e}")

class ManifoldUnitaryRecovery:
    """
    Wrapper for manual execution of the recovery process.
    """
    def __init__(self, model: nn.Module):
        self.hook = UnitaryRecoveryHook(model)

    def apply(self):
        self.hook.recover_manifold()

def calculate_spectral_shift(w_old, w_new):
    """
    Krein-like trace formula approximation for manifold transition.
    """
    # η = (1/π) arg{det(S)} approximated by the geodesic distance on the manifold
    return torch.dist(w_old, w_new).item()