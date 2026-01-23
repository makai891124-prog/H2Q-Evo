import torch
import torch.nn as nn
import torch.nn.functional as F

class ExampleLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Example loss: Cross-entropy loss
        loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        return loss
