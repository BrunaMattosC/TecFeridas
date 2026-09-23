"""
batch_inference.py
===================
Etapa 15 - Inferencia em lote.

Le todas as imagens de uma pasta, segmenta cada uma, salva a mascara
prevista, o overlay e a imagem anotada com as medidas, mede todas as
feridas encontradas e gera dois CSVs consolidados:

    - ``medicoes_lote.csv``      -> uma linha por imagem (ferida principal);
    - ``medicoes_por_ferida.csv`` -> uma linha por ferida detectada.

Uso:
    python batch_inference.py --input-dir caminho/das/imagens \
        --checkpoint checkpoints/best_xxx.pt --output-dir outputs/lote
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union

import cv2
import pandas as pd
from tqdm import tqdm

import config
from config import IMAGE_EXTENSIONS
from hardware import ajustar_backends, banner_hardware, resolver_dispositivo, texto_memoria_gpu
from measure import (
    PixelScale,
    adicionar_args_escala,
    annotate_measurements,
    clean_mask,
    measure_wound,
    measurements_to_row,
    measure_wounds,
    scale_from_args,
    wounds_to_rows,
)
from predict import load_model_from_checkpoint, predict_mask
from utils import get_logger, overlay_mask_on_image, save_image

logger = get_logger(__name__, config.LOGS_DIR / "batch_inference.log")


def run_batch_inference(
    input_dir: Path,
    checkpoint_path: Path,
    output_dir: Optional[Path] = None,
    pixels_por_cm: Union[PixelScale, float, None] = None,
) -> pd.DataFrame:
    """Segmenta e mede todas as imagens de uma pasta.

    Args:
        input_dir: pasta contendo as imagens a processar.
        checkpoint_path: checkpoint do modelo treinado.
        output_dir: pasta onde máscaras/overlays/CSV serão salvos; se
            ``None``, usa ``config.PREDICTIONS_DIR / "lote"``.
        pixels_por_cm: fator de escala opcional para medidas em cm
            (número em pixels/cm ou :class:`measure.PixelScale`).

    Returns:
        DataFrame com uma linha de medição por imagem processada
        (ferida principal). O detalhamento por ferida é salvo em
        ``medicoes_por_ferida.csv`` dentro de ``output_dir``.

    Raises:
        FileNotFoundError: se ``input_dir`` não existir ou não tiver imagens.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {input_dir}")

    image_paths: List[Path] = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {input_dir}")

    output_dir = output_dir or (config.PREDICTIONS_DIR / "lote")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mesma resolucao de dispositivo do treino/avaliacao. A inferencia em
    # lote nao exige GPU, mas avisa quando ha GPU disponivel e ociosa.
    device = resolver_dispositivo(exigir_gpu=False)
    ajustar_backends(device)
    print(banner_hardware(device))

    model = load_model_from_checkpoint(checkpoint_path, device)

    records = []
    wound_records = []
    for image_path in tqdm(image_paths, desc="inferência em lote"):
        try:
            image, mask = predict_mask(model, image_path, device)
        except Exception as exc:  # captura e prossegue com as demais imagens
            logger.error("Falha ao processar %s: %s", image_path.name, exc)
            continue

        # A máscara é gravada em tons de cinza (2-D), então não passa pela
        # conversão RGB->BGR de ``save_image``.
        cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.png"), mask * 255)
        overlay = overlay_mask_on_image(image, mask)
        save_image(output_dir / f"{image_path.stem}_overlay.png", overlay)

        measured_mask = clean_mask(mask)
        wounds = measure_wounds(measured_mask, pixels_por_cm=pixels_por_cm, cleanup=False)
        wound_records.extend(wounds_to_rows(image_path.name, wounds))

        annotated = annotate_measurements(image, wounds, mask=measured_mask)
        save_image(output_dir / f"{image_path.stem}_medicoes.png", annotated)

        try:
            measurements = measure_wound(mask, pixels_por_cm=pixels_por_cm)
            row = measurements_to_row(image_path.name, measurements, num_feridas=len(wounds))
        except ValueError:
            row = {"nome_imagem": image_path.name, "num_feridas": 0,
                   "observacao": "nenhuma ferida detectada"}
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = output_dir / "medicoes_lote.csv"
    df.to_csv(csv_path, index=False)

    wounds_csv_path = output_dir / "medicoes_por_ferida.csv"
    pd.DataFrame(wound_records).to_csv(wounds_csv_path, index=False)

    logger.info(
        "Inferência em lote concluída em %s: %d imagens processadas, %d feridas "
        "medidas. CSVs salvos em %s e %s",
        device, len(records), len(wound_records), csv_path, wounds_csv_path,
    )
    if device.type == "cuda":
        logger.info("GPU: %s", texto_memoria_gpu(device))
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferência em lote sobre uma pasta de imagens.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    adicionar_args_escala(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_batch_inference(
        input_dir=Path(args.input_dir),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        pixels_por_cm=scale_from_args(args),
    )
