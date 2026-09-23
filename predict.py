"""
predict.py
==========
Etapa 8 — Predição em uma única imagem.

Recebe uma imagem, gera a máscara prevista, sobrepõe sobre a imagem
original, mede cada ferida encontrada, salva o resultado e mostra a
comparação:
    imagem original | máscara real (se disponível) | máscara prevista |
    overlay | medições
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from dataset import get_eval_transforms
from hardware import (
    ajustar_backends,
    conferir_modelo_no_dispositivo,
    configuracao_amp,
    contexto_autocast,
    para_dispositivo,
    resolver_dispositivo,
)
from models import build_model
from utils import get_logger, ler_imagem, overlay_mask_on_image, save_image

logger = get_logger(__name__, config.LOGS_DIR / "predict.log")


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Carrega um modelo a partir de um checkpoint salvo por :mod:`train`.

    Args:
        checkpoint_path: caminho do arquivo ``.pt``.
        device: device onde o modelo será carregado.

    Returns:
        Modelo em modo de avaliação (``eval()``), pronto para inferência.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    from config import ModelConfig
    model_cfg = ModelConfig(**checkpoint["model_config"])

    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Os pesos vêm do checkpoint DEPOIS do .to(device); a conferência
    # garante que o load_state_dict não reintroduziu tensores em CPU.
    conferir_modelo_no_dispositivo(model, device, nome="modelo do checkpoint")
    logger.info("Modelo carregado de %s em %s (val_dice=%.4f)",
                checkpoint_path, device, checkpoint.get("val_dice", -1))
    return model


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    threshold: float = 0.5,
    image_size: int = config.DATA_CFG.image_size,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera a máscara binária prevista para uma imagem.

    Args:
        model: modelo de segmentação treinado.
        image_path: caminho da imagem de entrada.
        device: device de inferência.
        threshold: limiar de binarização da probabilidade.
        image_size: tamanho para o qual a imagem é redimensionada antes da inferência.

    Returns:
        Tupla ``(imagem_original_rgb, mascara_binaria_no_tamanho_original)``.
    """
    image_bgr = ler_imagem(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]  # (H, W)

    transform = get_eval_transforms(image_size)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0)
    transformed = para_dispositivo(tensor, device, nao_bloqueante=False)

    usar_amp, amp_dtype, _ = configuracao_amp(device)
    with contexto_autocast(device, usar_amp, amp_dtype):
        logits = model(transformed)

    # ``.float()`` antes do sigmoid mantém a máscara idêntica à de uma
    # inferência em fp32; ``.cpu()`` só acontece aqui, no fim, porque a
    # medição (OpenCV) é necessariamente feita em CPU.
    probs = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
    mask_small = (probs > threshold).astype(np.uint8)

    mask_full = cv2.resize(
        mask_small, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST
    )
    return image_rgb, mask_full


def visualize_prediction(
    image: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    annotated: Optional[np.ndarray] = None,
) -> None:
    """Monta e salva o grid de comparação: original | real | prevista | overlay.

    Args:
        image: imagem original RGB.
        predicted_mask: máscara binária prevista pelo modelo.
        ground_truth_mask: máscara real, se disponível (opcional).
        save_path: caminho para salvar a figura; se ``None``, não salva.
        annotated: imagem já anotada com as medições (opcional). Quando
            informada, um painel adicional "Medições" é acrescentado ao
            final do grid.
    """
    overlay = overlay_mask_on_image(image, predicted_mask)
    n_panels = 3 + (ground_truth_mask is not None) + (annotated is not None)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes = np.atleast_1d(axes)

    axes[0].imshow(image)
    axes[0].set_title("Imagem original")
    axes[0].axis("off")

    idx = 1
    if ground_truth_mask is not None:
        axes[idx].imshow(ground_truth_mask, cmap="gray")
        axes[idx].set_title("Máscara real")
        axes[idx].axis("off")
        idx += 1

    axes[idx].imshow(predicted_mask, cmap="gray")
    axes[idx].set_title("Máscara prevista")
    axes[idx].axis("off")
    idx += 1

    axes[idx].imshow(overlay)
    axes[idx].set_title("Overlay")
    axes[idx].axis("off")
    idx += 1

    if annotated is not None:
        axes[idx].imshow(annotated)
        axes[idx].set_title("Medições")
        axes[idx].axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Comparação visual salva em %s", save_path)
    plt.close(fig)


def predict_single_image(
    image_path: Path,
    checkpoint_path: Path,
    ground_truth_path: Optional[Path] = None,
    pixels_por_cm: Optional[float] = None,
    scale: Optional["PixelScale"] = None,
) -> dict:
    """Pipeline completo de predição para uma única imagem: segmenta,
    mede cada ferida e salva as visualizações.

    Fluxo: imagem -> segmentação -> máscara -> limpeza -> medição por
    ferida -> visualização anotada + CSV.

    Args:
        image_path: caminho da imagem a segmentar.
        checkpoint_path: checkpoint do modelo treinado.
        ground_truth_path: máscara real, se disponível, para comparação visual.
        pixels_por_cm: fator de escala opcional para medidas em cm.
        scale: :class:`measure.PixelScale` já construída; tem precedência
            sobre ``pixels_por_cm``.

    Returns:
        Dicionário com as medições da ferida principal (via
        :func:`measure.measure_wound`, mantendo o formato histórico),
        acrescido da chave ``"feridas"`` com uma linha por ferida
        detectada e de ``"num_feridas"``.
    """
    from dataclasses import asdict

    from measure import (
        annotate_measurements,
        clean_mask,
        format_measurements_text,
        measure_wound,
        measure_wounds,
        wounds_to_rows,
    )

    # Inferência em uma única imagem continua útil em CPU, então aqui a GPU
    # não é exigida — mas, se houver GPU ociosa, o aviso aparece do mesmo
    # jeito (nada de fallback silencioso).
    device = resolver_dispositivo(exigir_gpu=False)
    ajustar_backends(device)
    logger.info("Inferência em device: %s", device)

    model = load_model_from_checkpoint(checkpoint_path, device)

    image, predicted_mask = predict_mask(model, image_path, device)

    ground_truth = None
    if ground_truth_path is not None and ground_truth_path.exists():
        ground_truth = ler_imagem(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    effective_scale = scale if scale is not None else pixels_por_cm

    # A máscara medida (pós-limpeza) é também a exibida no painel de
    # medições, para que overlay e números sempre concordem. O overlay
    # legado continua usando a máscara crua do modelo.
    measured_mask = clean_mask(predicted_mask)
    wounds = measure_wounds(measured_mask, pixels_por_cm=effective_scale, cleanup=False)
    annotated = annotate_measurements(image, wounds, mask=measured_mask)

    save_path = config.PREDICTIONS_DIR / f"{image_path.stem}_comparacao.png"
    visualize_prediction(image, predicted_mask, ground_truth, save_path, annotated=annotated)

    overlay = overlay_mask_on_image(image, predicted_mask)
    save_image(config.PREDICTIONS_DIR / f"{image_path.stem}_overlay.png", overlay)
    save_image(config.PREDICTIONS_DIR / f"{image_path.stem}_medicoes.png", annotated)

    rows = wounds_to_rows(image_path.name, wounds)
    csv_path = config.PREDICTIONS_DIR / f"{image_path.stem}_medicoes.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    logger.info("Medições para %s:\n%s", image_path.name, format_measurements_text(wounds))
    logger.info("Medições salvas em %s", csv_path)

    if not wounds:
        logger.warning("Nenhuma ferida detectada em %s.", image_path.name)
        return {"nome_imagem": image_path.name, "num_feridas": 0, "feridas": rows}

    # Ferida principal (maior componente) — mantém o formato de retorno histórico.
    result = asdict(measure_wound(predicted_mask, pixels_por_cm=effective_scale))
    result["num_feridas"] = len(wounds)
    result["feridas"] = rows
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predição de segmentação de ferida em uma imagem.")
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem de entrada.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho do checkpoint do modelo.")
    parser.add_argument("--ground-truth", type=str, default=None, help="Máscara real (opcional).")
    parser.add_argument("--pixels-por-cm", type=float, default=None, help="Fator de escala pixels/cm (opcional).")
    parser.add_argument("--pixels-por-mm", type=float, default=None, help="Fator de escala pixels/mm (opcional).")
    parser.add_argument(
        "--referencia-pixels", type=float, default=None,
        help="Tamanho, em pixels, de um objeto de referência na imagem.",
    )
    parser.add_argument(
        "--referencia-cm", type=float, default=None,
        help="Tamanho real, em cm, do objeto de referência informado.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from measure import scale_from_options

    args = _parse_args()
    predict_single_image(
        image_path=Path(args.image),
        checkpoint_path=Path(args.checkpoint),
        ground_truth_path=Path(args.ground_truth) if args.ground_truth else None,
        scale=scale_from_options(
            pixels_por_cm=args.pixels_por_cm,
            pixels_por_mm=args.pixels_por_mm,
            referencia_pixels=args.referencia_pixels,
            referencia_cm=args.referencia_cm,
        ),
    )
