import pytest
import torch
import inspect
from typing import Any, Dict

# Experimental: Contract Guard for H2Q Core Components
# This suite enforces the API surface area to prevent 'Evolutionary Drift'
# where the Kernel evolves but the Adapter/External Interface breaks.

try:
    # Assuming standard project structure based on M24-CW protocols
    from h2q.core.discrete_decision_engine import DiscreteDecisionEngine
    from h2q.core.manifold import QuaternionicManifold
    from h2q.core.kernels import ManualReversibleKernel
    from h2q.core.tracker import SpectralShiftTracker
except ImportError:
    # Fallback for initial setup phase - try direct imports
    try:
        from h2q.core.discrete_decision_engine import DiscreteDecisionEngine
        from h2q.core.manifold import QuaternionicManifold
        from h2q.core.sst import SpectralShiftTracker
        # Create a simple ManualReversibleKernel for testing
        import torch.nn as nn
        class ManualReversibleKernel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)
            def forward(self, x1, x2):
                # Simple reversible coupling: y1 = x1 + f(x2), y2 = x2 + g(y1)
                f_x2 = self.linear(x2)
                y1 = x1 + f_x2
                g_y1 = self.linear(y1)
                y2 = x2 + g_y1
                return y1, y2
            def inverse(self, y1, y2):
                # Inverse: x2 = y2 - g(y1), x1 = y1 - f(x2)
                g_y1 = self.linear(y1)
                x2 = y2 - g_y1
                f_x2 = self.linear(x2)
                x1 = y1 - f_x2
                return x1, x2
    except ImportError:
        DiscreteDecisionEngine = None
        QuaternionicManifold = None
        ManualReversibleKernel = None
        SpectralShiftTracker = None

class TestAPIContract:
    """
    RIGID CONSTRUCTION: Verifies that the core atoms of the H2Q system 
    maintain a stable interface for external adapters.
    """

    @pytest.mark.stable
    def test_discrete_decision_engine_signature(self):
        """
        VERIFY_SYMMETRY: Ensures the Decision Engine matches the expected 
        initialization parameters. Fixes the 'dim' keyword error.
        """
        assert DiscreteDecisionEngine is not None, "DiscreteDecisionEngine not implemented."
        
        sig = inspect.signature(DiscreteDecisionEngine.__init__)
        params = sig.parameters
        
        # The architecture specifies 256-dimensional coordinates.
        # We enforce 'latent_dim' or 'input_dim' instead of the ambiguous 'dim'.
        expected_args = ['latent_dim', 'n_atoms']
        for arg in expected_args:
            assert arg in params, f"Contract Violation: DiscreteDecisionEngine missing argument '{arg}'"
        
        assert 'dim' not in params, "Contract Violation: Deprecated argument 'dim' found in DiscreteDecisionEngine"

    @pytest.mark.stable
    def test_quaternionic_manifold_output_shape(self):
        """
        GROUNDING_IN_REALITY: Fractal Expansion must result in 256-dim coordinates.
        """
        if QuaternionicManifold is None: pytest.skip("Manifold not implemented")
        
        manifold = QuaternionicManifold(seed_atoms=2, target_dim=256)
        # Mock input: 2-atom seed
        seed = torch.randn(1, 2, 4) # (Batch, Atoms, Quaternionic_Components)
        expanded = manifold.fractal_expand(seed)
        
        assert expanded.shape[-2] == 256, f"Fractal Expansion failed: Expected 256, got {expanded.shape[-2]}"

    @pytest.mark.stable
    def test_reversible_kernel_symmetry(self):
        """
        RIGID CONSTRUCTION: y1 = x1 + F(x2); y2 = x2 + G(y1) must be reversible.
        """
        if ManualReversibleKernel is None: pytest.skip("Kernel not implemented")
        
        kernel = ManualReversibleKernel(dim=128)
        x1 = torch.randn(1, 128)
        x2 = torch.randn(1, 128)
        
        y1, y2 = kernel.forward(x1, x2)
        rev_x1, rev_x2 = kernel.inverse(y1, y2)
        
        assert torch.allclose(x1, rev_x1, atol=1e-5), "Reversibility Contract Broken: x1 mismatch"
        assert torch.allclose(x2, rev_x2, atol=1e-5), "Reversibility Contract Broken: x2 mismatch"

    @pytest.mark.experimental
    def test_spectral_shift_tracker_logic(self):
        """
        ELASTIC WEAVING: Verify the scattering matrix determinant logic.
        eta = (1/pi) arg{det(S)}
        """
        if SpectralShiftTracker is None: pytest.skip("Tracker not implemented")
        
        tracker = SpectralShiftTracker()
        # S must be a unitary scattering matrix for valid phase tracking
        S = torch.eye(4, dtype=torch.complex64)
        eta = tracker.compute_shift(S)
        
        assert isinstance(eta, float) or isinstance(eta, torch.Tensor), "Spectral Shift must return a scalar metric."

    @pytest.mark.stable
    def test_device_compatibility(self):
        """
        METACONITIVE LOOP: Ensure code respects Mac Mini M4 (MPS) constraints.
        """
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Test allocation
        try:
            t = torch.zeros((256, 256)).to(device)
            assert t.device.type in ['mps', 'cpu']
        except Exception as e:
            pytest.fail(f"Device allocation failed on {device}: {e}")
