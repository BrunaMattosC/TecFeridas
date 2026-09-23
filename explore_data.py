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

import config
from dataset import explore_dataset, list_image_mask_pairs
from utils import get_logger, ler_imagem, overlay_mask_on_image, save_figure

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

    save_figure(fig, out_path, "Histograma de área", logger)


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

    save_figure(fig, out_path, "Grid de exemplos aleatórios", logger)


def main() -> None:
    # ``explore_dataset()`` já registra contagens e resolução média no log;
    # repeti-las aqui só duplicaria as mesmas linhas.
    df = explore_dataset()
    df.to_csv(config.REPORTS_DIR / "exploracao_dataset.csv", index=False)
    plot_area_distribution(df)
    plot_random_examples()


if __name__ == "__main__":
    main()
