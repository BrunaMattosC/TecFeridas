"""
losses.py
=========
Etapa 5 — Função de perda: Dice Loss + Binary Cross-Entropy.

A perda final é a combinação ponderada:

    Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice Loss para segmentação binária, aplicada sobre logits.

    Args:
        smooth: termo de suavização para evitar divisão por zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combinação ponderada de Dice Loss e BCE-with-logits.

    Args:
        dice_weight: peso da Dice Loss no total.
        bce_weight: peso da BCE no total.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)
