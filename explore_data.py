"""
explore_data.py
================
Etapa 2 — Exploração automática dos dados.

Gera:
    - contagem de imagens e máscaras;
    - resolução média;
    - histograma da área das feridas (em pixels e percentual);
    - grid de exemplos aleatórios (imagem + máscara sobreposta).

Uso:
    python explore_data.py
"""

from __future__ import annotations

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config
from dataset import contar_imagens, explore_dataset, list_image_mask_pairs
from utils import get_logger, ler_imagem, overlay_mask_on_image

logger = get_logger(__name__, config.LOGS_DIR / "explore_data.log")


def plot_area_distribution(df, out_path=config.FIGURES_DIR / "distribuicao_area.png") -> None:
    """Plota histogramas da área absoluta e percentual das feridas."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["area_ferida_pixels"], bins=40, color="steelblue")
    axes[0].set_title("Distribuição da área da ferida (pixels)")
    axes[0].set_xlabel("Área (pixels)")
    axes[0].set_ylabel("Frequência")

    axes[1].hist(df["percentual_area"], bins=40, color="indianred")
    axes[1].set_title("Distribuição do percentual de área ocupada")
    axes[1].set_xlabel("% da imagem")
    axes[1].set_ylabel("Frequência")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Histograma de área salvo em %s", out_path)


def plot_random_examples(n: int = 6, out_path=config.FIGURES_DIR / "exemplos_aleatorios.png") -> None:
    """Mostra ``n`` exemplos aleatórios (imagem original + overlay da máscara)."""
    pairs = list_image_mask_pairs()
    sample = random.sample(pairs, min(n, len(pairs)))

    fig, axes = plt.subplots(2, len(sample), figsize=(4 * len(sample), 8))
    for i, (img_path, mask_path) in enumerate(sample):
        image = cv2.cvtColor(ler_imagem(img_path), cv2.COLOR_BGR2RGB)
        mask = ler_imagem(mask_path, cv2.IMREAD_GRAYSCALE)
        overlay = overlay_mask_on_image(image, mask)

        axes[0, i].imshow(image)
        axes[0, i].set_title(img_path.name, fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title("máscara sobreposta", fontsize=8)
        axes[1, i].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Grid de exemplos aleatórios salvo em %s", out_path)


def main() -> None:
    df = explore_dataset()
    n_images = contar_imagens(config.IMAGES_DIR)
    n_masks = contar_imagens(config.MASKS_DIR)

    logger.info("Número de imagens: %d", n_images)
    logger.info("Número de máscaras: %d", n_masks)
    logger.info("Resolução média: %.0fx%.0f", df["altura"].mean(), df["largura"].mean())

    df.to_csv(config.REPORTS_DIR / "exploracao_dataset.csv", index=False)
    plot_area_distribution(df)
    plot_random_examples()


if __name__ == "__main__":
    main()
