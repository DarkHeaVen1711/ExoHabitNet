"""
losses.py
---------
Focal loss implementation and helpers for ExoHabitNet.

Usage:
    from utils.losses import FocalLoss

Supports optional per-class `weight` (same semantics as PyTorch's CrossEntropyLoss)
and an `alpha` scaling factor (scalar or per-class tensor) to emphasize classes.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        gamma: focusing parameter >= 0. Typical values: 1.0-3.0
        weight: optional class weight tensor (same as CrossEntropyLoss)
        alpha: optional scalar or Tensor(C,) to further scale loss per-class
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer('weight', weight) if isinstance(weight, torch.Tensor) else setattr(self, 'weight', weight)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (N, C), targets: (N,)
        logpt = F.log_softmax(inputs, dim=1)
        # per-sample cross-entropy (supports weight)
        ce = F.nll_loss(logpt, targets, weight=self.weight, reduction='none')

        # pt = exp(-ce) because ce = -log(pt)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:
            # alpha may be scalar or per-class tensor
            if isinstance(self.alpha, torch.Tensor):
                at = self.alpha[targets].to(inputs.device)
            else:
                at = torch.full_like(loss, float(self.alpha), device=inputs.device)
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
