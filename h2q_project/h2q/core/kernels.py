# h2q/core/kernels.py

import torch
import torch.nn as nn
from .discrete_decision_engine import DiscreteDecisionEngine

class ManualReversibleKernel(nn.Module):
    """
    Manual Reversible Kernel for H2Q Framework.
    Implements reversible transformations using discrete decision engines.
    """
    def __init__(self, dim=256):
        super().__init__()
        # Split dim for the two-stream reversible architecture
        half_dim = dim // 2
        self.f_engine = DiscreteDecisionEngine(dim=half_dim)
        self.g_engine = DiscreteDecisionEngine(dim=half_dim)

    def forward(self, x1, x2):
        # RIGID CONSTRUCTION: y1 = x1 + F(x2); y2 = x2 + G(y1)
        y1 = x1 + self.f_engine(x2.unsqueeze(0)).squeeze(0)
        y2 = x2 + self.g_engine(y1.unsqueeze(0)).squeeze(0)
        return y1, y2

    def inverse(self, y1, y2):
        # Reverse the transformation
        x2 = y2 - self.g_engine(y1.unsqueeze(0)).squeeze(0)
        x1 = y1 - self.f_engine(x2.unsqueeze(0)).squeeze(0)
        return x1, x2

__all__ = ['ManualReversibleKernel']