"""
teccurar_relatorio.py
======================
Relatório PDF de uma análise individual do TecCurar.

Recebe um :class:`teccurar_pipeline.ResultadoAnalise` e produz um PDF de
uma página com a fotografia original, a máscara segmentada, a sobreposição
das duas, as medidas em pixels e em centímetros, e a procedência da
calibração.

Fica separado da interface de propósito: quem gera o relatório não precisa
saber se a análise veio de uma janela gráfica ou de um script.

O PDF é montado com ``matplotlib``, que o projeto já usa para as figuras de
avaliação — nenhuma dependência nova é introduzida.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import config
from teccurar_pipeline import (
    CALIBRACAO_AUSENTE,
    ResultadoAnalise,
    linhas_de_medidas,
)
from utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR / "teccurar_relatorio.log")

RELATORIOS_DIR = config.BASE_DIR / "Resultados" / "Relatorios" / "Inferencia"


def nome_padrao(quando: Optional[datetime] = None) -> str:
    """Nome de arquivo com carimbo de tempo, para não sobrescrever nada."""
    quando = quando or datetime.now()
    return "relatorio_teccurar_%s.pdf" % quando.strftime("%Y%m%d_%H%M%S")


def gerar_relatorio(resultado: ResultadoAnalise, destino: Optional[Path] = None) -> Path:
    """Gera o PDF da análise e devolve o caminho do arquivo criado.

    Args:
        resultado: análise produzida por :func:`teccurar_pipeline.analisar_imagem`.
        destino: caminho do PDF; se ``None``, grava em
            ``Resultados/Relatorios/Inferencia`` com nome datado.

    Returns:
        Caminho do PDF gerado.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    destino = Path(destino) if destino else RELATORIOS_DIR / nome_padrao(resultado.quando)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(destino) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 retrato

        fig.text(0.5, 0.955, "TecCurar — Relatório de Análise de Ferida",
                 ha="center", size=15, weight="bold")
        fig.text(0.5, 0.930, resultado.caminho_imagem.name, ha="center",
                 size=9.5, color="#555555")

        # --- as três vistas obrigatórias ---------------------------------- #
        vistas = [
            ("Fotografia original", resultado.imagem_rgb),
            ("Máscara segmentada", np.dstack([resultado.mascara * 255] * 3)),
            ("Fotografia + máscara", resultado.sobreposicao),
        ]
        for i, (titulo, arte) in enumerate(vistas):
            eixo = fig.add_axes([0.055 + i * 0.315, 0.660, 0.285, 0.225])
            eixo.imshow(arte)
            eixo.axis("off")
            eixo.set_title(titulo, size=9)

        # --- medidas -------------------------------------------------------- #
        fig.text(0.055, 0.600, "Medidas da ferida", size=12, weight="bold")
        if len(resultado.feridas) > 1:
            fig.text(0.055, 0.578,
                     "Foram detectadas %d feridas; a tabela descreve a maior delas."
                     % len(resultado.feridas), size=8.5, color="#555555")

        linhas = [[nome, px, cm] for nome, px, cm in linhas_de_medidas(resultado)]
        eixo = fig.add_axes([0.055, 0.425, 0.89, 0.150])
        eixo.axis("off")
        tabela = eixo.table(cellText=linhas,
                            colLabels=["Medida", "Pixels", "Centímetros"],
                            loc="center", cellLoc="center",
                            colWidths=[0.34, 0.33, 0.33])
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)
        tabela.scale(1, 2.0)
        for j in range(3):
            tabela[0, j].set_facecolor("#3d5a73")
            tabela[0, j].set_text_props(color="white", weight="bold")
        for i in range(1, len(linhas) + 1):
            tabela[i, 0].set_text_props(ha="left")
            if not resultado.tem_centimetros:
                tabela[i, 2].set_facecolor("#f4f4f4")

        # --- calibração ----------------------------------------------------- #
        fig.text(0.055, 0.388, "Calibração", size=12, weight="bold")
        if resultado.tem_centimetros:
            escala = resultado.escala
            texto_cal = (
                "Situação   : conversão física disponível\n"
                "Referência : %s\n"
                "Escala     : %.3f pixels/cm   (%.6f cm/pixel)\n\n"
                "Medidas lineares convertidas por  cm = pixels × cm/pixel.\n"
                "Áreas convertidas por  cm² = pixels² × (cm/pixel)²."
                % (resultado.status_calibracao,
                   escala.pixels_per_cm, 1.0 / escala.pixels_per_cm)
            )
        else:
            texto_cal = (
                "Situação   : conversão física NÃO disponível\n"
                "Referência : nenhuma foi informada para esta fotografia\n\n"
                "As medidas acima estão apenas em pixels. Uma fotografia não\n"
                "carrega, por si só, informação de tamanho real: a mesma ferida\n"
                "ocupa mais ou menos pixels conforme a distância da câmera.\n\n"
                "Para obter centímetros, refaça a análise informando um fator\n"
                "pixels/cm conhecido, ou fotografando junto à ferida um objeto de\n"
                "tamanho conhecido — uma moeda, por exemplo — e informando seu\n"
                "tamanho em pixels e em centímetros."
            )
        fig.text(0.055, 0.205, texto_cal, size=8.8, va="bottom",
                 family="monospace", linespacing=1.6)

        # --- procedência ----------------------------------------------------- #
        fig.text(0.055, 0.155, "Procedência da análise", size=12, weight="bold")
        fig.text(0.055, 0.050, (
            "Data e hora : %s\n"
            "Arquivo     : %s\n"
            "Modelo      : %s\n"
            "Feridas     : %d detectada(s)\n"
            "Resolução   : %d x %d pixels"
            % (resultado.quando.strftime("%d/%m/%Y %H:%M:%S"),
               resultado.caminho_imagem.name,
               resultado.checkpoint.name,
               len(resultado.feridas),
               resultado.imagem_rgb.shape[1], resultado.imagem_rgb.shape[0])
        ), size=8.8, va="bottom", family="monospace", linespacing=1.6)

        fig.text(0.5, 0.018,
                 "Relatório gerado pelo TecCurar. As medidas descrevem a região "
                 "segmentada automaticamente e não substituem avaliação clínica.",
                 ha="center", size=7.5, color="#777777")

        pdf.savefig(fig)
        plt.close(fig)

    logger.info("Relatorio gerado em %s", destino)
    return destino
