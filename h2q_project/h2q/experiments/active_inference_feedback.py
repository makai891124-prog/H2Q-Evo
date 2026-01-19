import torch
import torch.nn as nn
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine
from h2q.core.sst import SpectralShiftTracker
from h2q.fdc_kernel import continuous_environment_drag

class ActiveInferenceFeedbackLoop:
    """
    Implements the closed-loop feedback mechanism where environmental drag mu(E)
    directly modulates the Discrete Decision Engine's (DDE) exploration alpha.
    """
    def __init__(self, dim=256, initial_alpha=0.1, device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        # Initialize Core H2Q Components
        self.dde = DiscreteDecisionEngine(dim=dim, alpha=initial_alpha).to(self.device)
        self.sst = SpectralShiftTracker(dim=dim).to(self.device)
        
        # State tracking
        self.current_eta = 0.0
        self.drag_history = []

    def compute_feedback_step(self, state_tensor, environment_noise):
        """
        Executes one iteration of the Active Inference loop.
        1. Calculate environmental drag mu(E).
        2. Modulate DDE alpha based on drag.
        3. Select geodesic path via DDE.
        4. Update Spectral Shift (eta).
        """
        # 0.1 No Deception: Ensure input is on correct device
        state_tensor = state_tensor.to(self.device)
        
        # 1. Calculate Environmental Drag mu(E)
        # In H2Q, drag represents the non-analytic resistance of the manifold
        mu_e = continuous_environment_drag(state_tensor, environment_noise)
        
        # 2. Modulate DDE Alpha (Exploration vs Exploitation)
        # High drag (uncertainty/noise) increases alpha to encourage exploration of stable geodesics
        # We use a clamped linear modulation to maintain stability
        new_alpha = torch.clamp(mu_e * 2.0, min=0.01, max=1.0)
        self.dde.alpha = new_alpha.item()
        
        # 3. Decision Phase
        # DDE selects the optimal discrete action/path based on the modulated alpha
        decision, log_probs = self.dde(state_tensor)
        
        # 4. Update Spectral Shift Tracker (eta)
        # eta = (1/pi) arg{det(S)} - tracking the accumulated Berry Phase
        self.current_eta = self.sst.update(state_tensor, decision)
        
        return {
            "decision": decision,
            "mu_e": mu_e.item(),
            "alpha": self.dde.alpha,
            "eta": self.current_eta
        }

if __name__ == "__main__":
    # Experimental validation on M4 Silicon
    print("[M24-CW] Initializing Active Inference Feedback Loop...")
    loop = ActiveInferenceFeedbackLoop(dim=256)
    
    # Simulate 10 steps of environmental interaction
    for i in range(10):
        mock_state = torch.randn(1, 256)
        mock_noise = torch.rand(1) * 0.5
        
        metrics = loop.compute_feedback_step(mock_state, mock_noise)
        
        print(f"Step {i} | Drag: {metrics['mu_e']:.4f} | Alpha: {metrics['alpha']:.4f} | Eta: {metrics['eta']:.4f}")
