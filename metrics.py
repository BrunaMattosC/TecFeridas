"""
metrics.py
==========
Etapa 6 — Métricas de segmentação calculadas durante treino/validação/teste:
Dice, IoU, Precision, Recall, F1 e Accuracy.

Implementadas com TorchMetrics para robustez numérica e consistência
com o ecossistema PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torchmetrics import Accuracy, JaccardIndex, Precision, Recall
from torchmetrics.classification import BinaryF1Score


@dataclass
class SegmentationMetrics:
    """Agrupa as métricas binárias de segmentação em um único objeto,
    todas movidas para o mesmo ``device`` do modelo.
    """

    device: torch.device
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.iou = JaccardIndex(task="binary", threshold=self.threshold).to(self.device)
        self.precision = Precision(task="binary", threshold=self.threshold).to(self.device)
        self.recall = Recall(task="binary", threshold=self.threshold).to(self.device)
        self.f1 = BinaryF1Score(threshold=self.threshold).to(self.device)
        self.accuracy = Accuracy(task="binary", threshold=self.threshold).to(self.device)

    def reset(self) -> None:
        for metric in (self.iou, self.precision, self.recall, self.f1, self.accuracy):
            metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        targets_int = targets.int()
        self.iou.update(probs, targets_int)
        self.precision.update(probs, targets_int)
        self.recall.update(probs, targets_int)
        self.f1.update(probs, targets_int)
        self.accuracy.update(probs, targets_int)

    def compute(self) -> Dict[str, float]:
        return {
            "iou": self.iou.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1": self.f1.compute().item(),
            "accuracy": self.accuracy.compute().item(),
        }


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Calcula o coeficiente de Dice para um lote de predições.

    Args:
        logits: saída crua (pré-sigmoid) do modelo, shape (N, 1, H, W).
        targets: máscaras verdadeiras binárias, mesmo shape de ``logits``.
        threshold: limiar para binarizar as probabilidades.
        smooth: termo de suavização.

    Returns:
        Valor escalar do Dice médio do lote.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_flat = preds.reshape(preds.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()
