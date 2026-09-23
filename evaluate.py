"""
evaluate.py
===========
Etapa 12 e 13 - Avaliacao completa do modelo no conjunto de teste e
visualizacoes agregadas.

Gera:
    - matriz de confusao;
    - IoU medio, Dice medio, Precision, Recall, F1, Accuracy;
    - curva ROC e curva Precision-Recall;
    - relatorio por imagem (area, perimetro, circularidade, IoU, Dice...)
      salvo em CSV (reaproveita measure.py / Etapa 11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import WoundDataset, get_eval_transforms, split_dataset
from hardware import (
    ajustar_backends,
    banner_hardware,
    conferir_lote_no_dispositivo,
    configuracao_amp,
    contexto_autocast,
    para_dispositivo,
    parametros_dataloader,
    resolver_dispositivo,
    texto_memoria_gpu,
)
from measure import measure_wound, measure_wounds, measurements_to_row, wounds_to_rows
from metrics import SegmentationMetrics, dice_coefficient
from predict import load_model_from_checkpoint
from utils import get_logger, save_figure

logger = get_logger(__name__, config.LOGS_DIR / "evaluate.log")


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Roda o modelo sobre todo o loader e coleta probabilidades/rotulos
    achatados (para ROC/PR/matriz de confusao) e metricas por imagem.

    Returns:
        Tupla (y_true_flat, y_prob_flat, per_image_records,
        per_wound_records, metricas_agregadas).
    """
    model.eval()
    metrics = SegmentationMetrics(device)
    metrics.reset()
    usar_amp, amp_dtype, _ = configuracao_amp(device)

    y_true_chunks, y_prob_chunks = [], []
    per_image_records = []
    per_wound_records = []

    primeiro = True
    for images, masks, names in tqdm(loader, desc="avaliacao final"):
        images = para_dispositivo(images, device)
        masks = para_dispositivo(masks, device)
        if primeiro:
            conferir_lote_no_dispositivo(device, imagens=images, mascaras=masks)
            primeiro = False

        with contexto_autocast(device, usar_amp, amp_dtype):
            logits = model(images)
        # Promove a float32 antes de sigmoid/metricas: as curvas ROC/PR e as
        # metricas ficam numericamente identicas a uma avaliacao em fp32.
        logits = logits.float()
        probs = torch.sigmoid(logits)

        metrics.update(logits, masks)

        # Subamostra os pixels para manter as curvas ROC/PR tratáveis em memória
        y_true_chunks.append(masks.cpu().numpy().reshape(-1)[::37])
        y_prob_chunks.append(probs.cpu().numpy().reshape(-1)[::37])

        for i, name in enumerate(names):
            pred_mask = (probs[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
            dice_i = dice_coefficient(logits[i : i + 1], masks[i : i + 1])

            wounds = measure_wounds(pred_mask)
            per_wound_records.extend(wounds_to_rows(name, wounds, dice=dice_i))

            try:
                measurements = measure_wound(pred_mask)
                row = measurements_to_row(name, measurements, dice=dice_i, num_feridas=len(wounds))
            except ValueError:
                row = {"nome_imagem": name, "dice": dice_i, "num_feridas": 0,
                       "observacao": "sem ferida detectada"}
            per_image_records.append(row)

    y_true = np.concatenate(y_true_chunks)
    y_prob = np.concatenate(y_prob_chunks)
    return y_true, y_prob, per_image_records, per_wound_records, metrics.compute()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Plota e salva a matriz de confusao binaria (pixel a pixel)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["fundo", "ferida"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusao (pixel-wise)")
    save_figure(fig, out_path, "Matriz de confusao", logger)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    """Plota e salva a curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # sklearn.metrics.auc faz a mesma integracao trapezoidal do antigo
    # np.trapz, que foi REMOVIDO no NumPy 2.0. Usar o sklearn (ja exigido
    # pelo projeto) mantem o resultado identico e torna o modulo imune a
    # versao do NumPy instalada.
    area_sob_curva = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {area_sob_curva:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curva ROC")
    ax.legend()
    save_figure(fig, out_path, "Curva ROC", logger)


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    """Plota e salva a curva Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    save_figure(fig, out_path, "Curva Precision-Recall", logger)


def evaluate_model(checkpoint_path: Path) -> Dict[str, float]:
    """Executa a avaliacao completa do modelo no conjunto de teste.

    Args:
        checkpoint_path: caminho do checkpoint treinado a avaliar.

    Returns:
        Dicionario com as metricas agregadas (iou, precision, recall, f1, accuracy).
    """
    # Mesmo dispositivo do treino, resolvido pela mesma funcao. A avaliacao
    # nao exige GPU (roda em CPU se preciso), mas avisa em voz alta se a GPU
    # existir e nao estiver sendo usada.
    device = resolver_dispositivo(exigir_gpu=False)
    ajustar_backends(device)
    print(banner_hardware(device))

    model = load_model_from_checkpoint(checkpoint_path, device)

    splits = split_dataset()
    test_ds = WoundDataset(splits.test, transform=get_eval_transforms())
    test_loader = DataLoader(
        test_ds, shuffle=False, **parametros_dataloader(device, treino=False)
    )

    y_true, y_prob, per_image_records, per_wound_records, agg_metrics = collect_predictions(
        model, test_loader, device
    )
    y_pred = (y_prob > 0.5).astype(np.uint8)

    plot_confusion_matrix(y_true, y_pred, config.FIGURES_DIR / "matriz_confusao.png")
    plot_roc_curve(y_true, y_prob, config.FIGURES_DIR / "curva_roc.png")
    plot_precision_recall_curve(y_true, y_prob, config.FIGURES_DIR / "curva_precision_recall.png")

    if per_image_records:
        report_df = pd.DataFrame(per_image_records)
        report_path = config.REPORTS_DIR / "relatorio_por_imagem.csv"
        report_df.to_csv(report_path, index=False)
        logger.info("Relatorio por imagem salvo em %s", report_path)

    if per_wound_records:
        wounds_path = config.REPORTS_DIR / "relatorio_por_ferida.csv"
        pd.DataFrame(per_wound_records).to_csv(wounds_path, index=False)
        logger.info("Relatorio por ferida salvo em %s", wounds_path)

    logger.info("Metricas agregadas no teste: %s", agg_metrics)
    if device.type == "cuda":
        logger.info("Avaliacao executada em %s | GPU: %s",
                    torch.cuda.get_device_name(device), texto_memoria_gpu(device))
    return agg_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avaliacao do modelo no conjunto de teste.")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate_model(Path(args.checkpoint))
