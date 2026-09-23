"""
gerar_mascaras_kaggle.py
=========================
Gera máscaras para as imagens do dataset Kaggle usando um modelo treinado.

Feito para o fluxo "treinar nas anotações da TecCurar e inferir sobre o
Kaggle": o checkpoint vem de ``train.py`` executado sobre
``dataset_teccurar/``, e aqui ele é aplicado às imagens de
``dataset/images``.

As máscaras existentes em ``dataset/masks`` **não são tocadas**. A saída
vai para uma pasta separada, de modo que o ground truth original continue
disponível para comparação e o pipeline padrão siga funcionando como antes.

Quando o ground truth existe, o script também mede a concordância entre a
máscara gerada e a original (Dice e IoU por imagem). Esse número é o que
diz se o resultado é aproveitável — sem ele, não há como saber se o modelo
aprendeu a segmentar ferida ou apenas a pintar manchas.

Uso:
    python gerar_mascaras_kaggle.py --checkpoint checkpoints/best_xxx.pt
    python gerar_mascaras_kaggle.py --checkpoint ... --limite 200   # amostra
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config
from hardware import ajustar_backends, banner_hardware, resolver_dispositivo
from predict import load_model_from_checkpoint, predict_mask
from utils import escrever_imagem, get_logger, ler_imagem

logger = get_logger(__name__, config.LOGS_DIR / "gerar_mascaras_kaggle.log")

SAIDA_PADRAO = config.DATASET_DIR / "masks_teccurar"
RELATORIO = config.REPORTS_DIR / "mascaras_kaggle_teccurar.csv"


def _dice_iou(gerada: np.ndarray, referencia: np.ndarray) -> tuple[float, float]:
    """Dice e IoU entre duas máscaras binárias de mesmo shape."""
    a = gerada > 0
    b = referencia > 0
    intersecao = float(np.logical_and(a, b).sum())
    soma = float(a.sum() + b.sum())
    uniao = float(np.logical_or(a, b).sum())
    dice = (2.0 * intersecao / soma) if soma else 1.0
    iou = (intersecao / uniao) if uniao else 1.0
    return dice, iou


def gerar(
    checkpoint_path: Path,
    images_dir: Optional[Path] = None,
    masks_ref_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    limite: Optional[int] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Roda o modelo sobre as imagens e grava uma máscara PNG para cada uma.

    Args:
        checkpoint_path: checkpoint treinado a aplicar.
        images_dir: imagens de entrada; ``None`` usa ``config.IMAGES_DIR``.
        masks_ref_dir: máscaras de referência para comparação; ``None`` usa
            ``config.MASKS_DIR``. Ausentes, a comparação é omitida.
        output_dir: destino das máscaras geradas; ``None`` usa
            ``dataset/masks_teccurar``.
        limite: processa apenas as N primeiras imagens (amostragem).
        threshold: limiar de binarização da probabilidade.

    Returns:
        Dicionário com as métricas agregadas da comparação.
    """
    images_dir = images_dir or config.IMAGES_DIR
    masks_ref_dir = masks_ref_dir or config.MASKS_DIR
    output_dir = output_dir or SAIDA_PADRAO
    output_dir.mkdir(parents=True, exist_ok=True)

    imagens = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")
    if limite:
        imagens = imagens[:limite]

    referencias = {
        p.stem.lower(): p for p in masks_ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    } if masks_ref_dir.exists() else {}

    # A geração de máscaras não exige GPU, mas avisa se houver uma ociosa.
    device = resolver_dispositivo(exigir_gpu=False)
    ajustar_backends(device)
    print(banner_hardware(device))

    model = load_model_from_checkpoint(checkpoint_path, device)

    linhas: List[dict] = []
    vazias = 0
    falhas: List[str] = []

    for caminho in tqdm(imagens, desc="gerando máscaras"):
        try:
            _, mascara = predict_mask(model, caminho, device, threshold=threshold)
        except Exception as exc:  # captura e prossegue com as demais imagens
            logger.error("Falha em %s: %s", caminho.name, exc)
            falhas.append(caminho.name)
            continue

        destino = output_dir / f"{caminho.stem}.png"
        escrever_imagem(destino, (mascara * 255).astype(np.uint8))

        area = float((mascara > 0).mean())
        if area == 0.0:
            vazias += 1

        linha = {
            "nome_imagem": caminho.name,
            "area_gerada_px": int((mascara > 0).sum()),
            "area_gerada_percent": round(100 * area, 3),
        }

        ref_path = referencias.get(caminho.stem.lower())
        if ref_path is not None:
            ref = ler_imagem(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref is not None:
                if ref.shape != mascara.shape:
                    ref = cv2.resize(ref, (mascara.shape[1], mascara.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                dice, iou = _dice_iou(mascara, ref > 127)
                linha["dice_vs_original"] = round(dice, 4)
                linha["iou_vs_original"] = round(iou, 4)
                linha["area_original_percent"] = round(100 * float((ref > 127).mean()), 3)
        linhas.append(linha)

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    with open(RELATORIO, "w", newline="", encoding="utf-8") as fh:
        campos = sorted({chave for linha in linhas for chave in linha})
        escritor = csv.DictWriter(fh, fieldnames=["nome_imagem"] + [c for c in campos if c != "nome_imagem"])
        escritor.writeheader()
        escritor.writerows(linhas)

    dices = [linha["dice_vs_original"] for linha in linhas if "dice_vs_original" in linha]
    ious = [linha["iou_vs_original"] for linha in linhas if "iou_vs_original" in linha]
    resumo = {
        "imagens": len(imagens),
        "geradas": len(linhas),
        "falhas": len(falhas),
        "mascaras_vazias": vazias,
        "comparadas": len(dices),
        "dice_medio": float(np.mean(dices)) if dices else float("nan"),
        "dice_mediano": float(np.median(dices)) if dices else float("nan"),
        "iou_medio": float(np.mean(ious)) if ious else float("nan"),
    }
    _relatar(resumo, dices, output_dir, falhas)
    return resumo


def _relatar(resumo, dices, output_dir, falhas) -> None:
    """Imprime o resumo da geração e da comparação contra o ground truth."""
    sep = "=" * 66
    print()
    print(sep)
    print("GERACAO DE MASCARAS -- modelo treinado na TecCurar")
    print(sep)
    print("Imagens processadas        : %d" % resumo["imagens"])
    print("Mascaras geradas           : %d" % resumo["geradas"])
    print("Falhas                     : %d" % resumo["falhas"])
    print("Mascaras COMPLETAMENTE vazias: %d (%.1f%%)" % (
        resumo["mascaras_vazias"],
        100 * resumo["mascaras_vazias"] / max(resumo["geradas"], 1)))
    if dices:
        arr = np.array(dices)
        print()
        print("Concordancia com o ground truth do Kaggle (%d imagens):" % len(dices))
        print("  Dice  medio=%.4f  mediano=%.4f" % (resumo["dice_medio"], resumo["dice_mediano"]))
        print("  IoU   medio=%.4f" % resumo["iou_medio"])
        print("  Dice < 0,10 : %d imagens (%.1f%%)" % ((arr < 0.10).sum(), 100*(arr < 0.10).mean()))
        print("  Dice > 0,50 : %d imagens (%.1f%%)" % ((arr > 0.50).sum(), 100*(arr > 0.50).mean()))
        print("  Dice > 0,70 : %d imagens (%.1f%%)" % ((arr > 0.70).sum(), 100*(arr > 0.70).mean()))
    if falhas:
        print()
        print("FALHAS (%d): %s" % (len(falhas), ", ".join(falhas[:10])))
    print()
    print("Mascaras geradas em : %s" % output_dir)
    print("Relatorio por imagem: %s" % RELATORIO)
    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera máscaras para o dataset Kaggle com um modelo treinado."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--masks-ref-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--limite", type=int, default=None,
                        help="Processa apenas as N primeiras imagens.")
    args = parser.parse_args()

    gerar(
        checkpoint_path=Path(args.checkpoint),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        masks_ref_dir=Path(args.masks_ref_dir) if args.masks_ref_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limite=args.limite,
    )
