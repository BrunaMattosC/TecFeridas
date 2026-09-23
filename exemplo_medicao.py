"""
exemplo_medicao.py
==================
Demonstracao da etapa de medicao SEM necessidade de modelo treinado.

A medicao (:mod:`measure`) opera sobre qualquer mascara binaria -- inclusive
as mascaras verdadeiras (ground truth) do proprio dataset. Isso permite ver
o sistema funcionando ponta a ponta antes de treinar qualquer coisa, e
tambem isola a qualidade da MEDICAO da qualidade da SEGMENTACAO.

Nao exige torch nem albumentations: bastam opencv, numpy e pandas.

Uso:
    python exemplo_medicao.py                    # escolhe uma imagem automaticamente
    python exemplo_medicao.py medetec_0153.png   # imagem especifica do dataset
    python exemplo_medicao.py --sem-rotulos      # util quando ha muitas feridas

Saidas em outputs/predictions/exemplo_medicao/:
    <nome>_medicoes.png  -- imagem anotada com a geometria e os valores
    <nome>_medicoes.csv  -- uma linha por ferida
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

import config
from config import IMAGE_EXTENSIONS
from measure import (
    annotate_measurements,
    clean_mask,
    format_measurements_text,
    measure_wounds,
    wounds_to_rows,
)
from utils import save_image


def escolher_imagem(nome: str | None) -> str:
    """Devolve o nome do arquivo a medir, validando que o par existe."""
    disponiveis = sorted(
        p.name for p in config.IMAGES_DIR.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not disponiveis:
        print("ERRO: nenhuma imagem em %s" % config.IMAGES_DIR)
        print("Baixe o dataset primeiro:  python dataset.py")
        raise SystemExit(1)

    if nome is None:
        nome = disponiveis[0]
        print("Nenhuma imagem informada; usando a primeira: %s" % nome)

    if not (config.IMAGES_DIR / nome).exists():
        print("ERRO: imagem nao encontrada: %s" % (config.IMAGES_DIR / nome))
        print("Exemplos disponiveis: %s" % ", ".join(disponiveis[:5]))
        raise SystemExit(1)
    if not (config.MASKS_DIR / nome).exists():
        print("ERRO: mascara correspondente nao encontrada: %s" % (config.MASKS_DIR / nome))
        raise SystemExit(1)
    return nome


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mede as feridas de uma mascara do dataset, sem modelo treinado."
    )
    parser.add_argument("imagem", nargs="?", default=None,
                        help="Nome do arquivo em dataset/images (ex.: medetec_0153.png).")
    parser.add_argument("--sem-rotulos", action="store_true",
                        help="Desenha so a geometria. Util quando ha muitas feridas, "
                             "porque os blocos de texto se sobrepoem.")
    args = parser.parse_args()

    nome = escolher_imagem(args.imagem)

    imagem = cv2.imread(str(config.IMAGES_DIR / nome))
    if imagem is None:
        print("ERRO: falha ao ler a imagem %s" % nome)
        return 1
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    mascara = cv2.imread(str(config.MASKS_DIR / nome), cv2.IMREAD_GRAYSCALE)

    # Limpa UMA vez e reusa a mesma mascara na medicao e no desenho, para
    # que os numeros do CSV e a imagem anotada nunca discordem.
    limpa = clean_mask(mascara)
    feridas = measure_wounds(limpa, cleanup=False)

    print()
    print("=" * 60)
    print("Imagem : %s" % nome)
    print("Feridas: %d" % len(feridas))
    print("=" * 60)
    print(format_measurements_text(feridas))

    saida = config.PREDICTIONS_DIR / "exemplo_medicao"
    saida.mkdir(parents=True, exist_ok=True)

    anotada = annotate_measurements(
        imagem, feridas, mask=limpa, draw_labels=not args.sem_rotulos
    )
    caminho_png = saida / ("%s_medicoes.png" % Path(nome).stem)
    save_image(caminho_png, anotada)

    caminho_csv = saida / ("%s_medicoes.csv" % Path(nome).stem)
    pd.DataFrame(wounds_to_rows(nome, feridas)).to_csv(caminho_csv, index=False)

    print()
    print("Imagem anotada: %s" % caminho_png)
    print("Relatorio CSV : %s" % caminho_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
