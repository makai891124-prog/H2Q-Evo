import torch
import torch.nn as nn
import torch.optim as optim
import math
from h2q.core.sst import SpectralShiftTracker
from h2q.core.optimizers.hjb_solver import get_hjb_solver
from h2q.core.discrete_decision_engine import get_canonical_dde

class HomeostaticTrainer:
    """
    Homeostatic Trainer: Orchestrates the transition between Wake (SGD) and 
    Sleep (HJB Geodesic Healing) based on the Heat-Death Index (HDI).
    
    HDI is modeled as the Von Neumann entropy of the singular value spectrum 
    of the manifold scattering matrix.
    """
    def __init__(self, model, optimizer, hdi_threshold=0.65, device="mps"):
        self.model = model
        self.optimizer = optimizer
        self.hdi_threshold = hdi_threshold
        self.device = device
        
        # Core H2Q Components
        self.sst = SpectralShiftTracker()
        self.hjb_solver = get_hjb_solver()
        # Using factory method to avoid 'dim' keyword argument error
        self.dde = get_canonical_dde()
        
        self.mode = "WAKE"
        self.hdi_history = []

    def calculate_hdi(self):
        """
        Calculates the Heat-Death Index (HDI) using Von Neumann entropy 
        of the singular value spectrum of the model's primary manifold weights.
        """
        total_entropy = 0.0
        count = 0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    # Flatten to 2D for SVD
                    w = param.view(param.size(0), -1)
                    # Use CPU for SVD if MPS stability is an issue, but try MPS first
                    try:
                        _, s, _ = torch.svd(w)
                    except:
                        _, s, _ = torch.svd(w.cpu())
                        s = s.to(self.device)
                    
                    # Normalize singular values to form a probability distribution
                    probs = (s**2) / (torch.sum(s**2) + 1e-9)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9))
                    # Normalize by max possible entropy (log of dimension)
                    max_entropy = math.log(len(s))
                    total_entropy += (entropy / max_entropy).item()
                    count += 1
        
        return total_entropy / count if count > 0 else 0.0

    def wake_step(self, data, target):
        """Standard SGD-based learning (Wake Phase)"""
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()
        
        # Update Spectral Shift Tracker
        self.sst.update(loss.item())
        return loss.item()

    def sleep_step(self):
        """HJB-based Geodesic Healing (Sleep Phase)"""
        self.model.eval()
        # The HJB solver adjusts weights to minimize the Fueter residual 
        # and restore manifold orthogonality (Geodesic Flow).
        healing_report = self.hjb_solver.solve(self.model)
        
        # Reset SST to reflect the new manifold state
        self.sst.reset_drift()
        return healing_report

    def step(self, data, target):
        """
        Orchestrates a single training iteration with homeostatic switching.
        """
        current_hdi = self.calculate_hdi()
        self.hdi_history.append(current_hdi)

        # Decision logic via DDE
        # If HDI exceeds threshold, the system is 'overheated' (high entropy)
        if current_hdi > self.hdi_threshold:
            self.mode = "SLEEP"
            report = self.sleep_step()
            return {"mode": "SLEEP", "hdi": current_hdi, "report": report}
        else:
            self.mode = "WAKE"
            loss = self.wake_step(data, target)
            return {"mode": "WAKE", "hdi": current_hdi, "loss": loss}

    def get_telemetry(self):
        return {
            "mode": self.mode,
            "hdi": self.hdi_history[-1] if self.hdi_history else 0.0,
            "spectral_shift": self.sst.get_shift()
        }