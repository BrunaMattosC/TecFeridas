"""
models.py
=========
Etapa 4 — Fábrica de modelos de segmentação usando ``segmentation_models_pytorch``.

Suporta as arquiteturas UNet (padrão), UNet++, DeepLabV3+ e FPN, com
backbones ResNet34, EfficientNet-B0 e EfficientNet-B3, todos
selecionáveis via :mod:`config`.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

from config import ModelConfig

_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
}

_SUPPORTED_ENCODERS = {"resnet34", "efficientnet-b0", "efficientnet-b3"}


def build_model(cfg: ModelConfig) -> nn.Module:
    """Constrói um modelo de segmentação a partir da configuração informada.

    Args:
        cfg: configuração do modelo (arquitetura, encoder, pesos, canais).

    Returns:
        Instância de ``nn.Module`` pronta para treino/inferência.

    Raises:
        ValueError: se a arquitetura ou o encoder solicitado não forem suportados.
    """
    arch_key = cfg.architecture.lower()
    if arch_key not in _ARCHITECTURES:
        raise ValueError(
            f"Arquitetura '{cfg.architecture}' não suportada. "
            f"Opções válidas: {sorted(_ARCHITECTURES)}"
        )

    if cfg.encoder_name not in _SUPPORTED_ENCODERS:
        raise ValueError(
            f"Encoder '{cfg.encoder_name}' não suportado. "
            f"Opções recomendadas: {sorted(_SUPPORTED_ENCODERS)}"
        )

    model_cls = _ARCHITECTURES[arch_key]
    model = model_cls(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
        activation=cfg.activation,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Conta o número de parâmetros treináveis de um modelo.

    Args:
        model: modelo PyTorch.

    Returns:
        Número total de parâmetros com ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
