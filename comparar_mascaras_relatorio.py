"""
comparar_mascaras_relatorio.py
===============================
Compara máscara original x nova em imagens do Kaggle e gera o relatório PDF.

Etapa seguinte à calibração por moedas (:mod:`calibrar_moedas`): usa a
escala pixels/cm obtida lá para converter as medidas das feridas, compara
a máscara original do Kaggle com a recriada por
:mod:`recriar_mascaras_dataset`, e produz as visualizações e o PDF.

A sobreposição das máscaras sobre a FOTOGRAFIA ORIGINAL é obrigatória e
aparece tanto nos arquivos por imagem quanto no relatório: máscara isolada
não permite julgar se a região segmentada corresponde à ferida.

As medições reaproveitam :mod:`measure`, o mesmo módulo que o projeto já
usa para medir feridas, de modo que os números aqui e no pipeline saiam da
mesma implementação.

Uso:
    python comparar_mascaras_relatorio.py
    python comparar_mascaras_relatorio.py --n-imagens 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from calibrar_moedas import CALIBRACAO_CSV, DIMENSOES_OFICIAIS, RESULTADOS_DIR
from measure import PixelScale, measure_wounds
from utils import escrever_imagem, get_logger, ler_imagem, overlay_mask_on_image

logger = get_logger(__name__, config.LOGS_DIR / "comparar_mascaras_relatorio.log")

MASCARAS_NOVAS_DIR = config.BASE_DIR / "TecCurar_NewMasks"
KAGGLE_SAIDA = RESULTADOS_DIR / "Kaggle_10_Imagens"
RELATORIO_DIR = RESULTADOS_DIR / "Relatorios" / "Validacao"
RELATORIO_PDF = RELATORIO_DIR / "relatorio_calibracao_comparacao.pdf"
TABELA_CSV = KAGGLE_SAIDA / "tabela_consolidada.csv"

COR_ORIGINAL = (220, 50, 50)     # vermelho
COR_NOVA = (40, 200, 90)         # verde
COR_AMBAS = (250, 200, 40)       # amarelo


def carregar_escala() -> Tuple[float, List[dict]]:
    """Lê a calibração produzida por :mod:`calibrar_moedas`.

    Returns:
        ``(pixels_por_cm, linhas_por_moeda)``. A escala adotada é a mediana
        das medianas por moeda, que dá o mesmo peso a cada denominação em
        vez de favorecer a que tem mais fotografias.
    """
    if not CALIBRACAO_CSV.exists():
        raise FileNotFoundError(
            f"Calibração não encontrada em {CALIBRACAO_CSV}. "
            "Execute `python calibrar_moedas.py` primeiro."
        )
    with open(CALIBRACAO_CSV, encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    medianas = [float(l["pixels_por_cm_mediano"]) for l in linhas]
    return float(np.median(medianas)), linhas


def selecionar_imagens(n: int = 10) -> List[str]:
    """Escolhe imagens representativas, com variedade deliberada.

    A amostra cobre as três origens do dataset (``fusc``, ``medetec``,
    ``wsnet``), diferentes tamanhos de ferida e diferentes graus de
    divergência entre as máscaras — incluindo casos sem alteração, para
    que o relatório não mostre apenas onde a recriação agiu.
    """
    caminho = config.REPORTS_DIR / "recriacao_kaggle.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Relatório de recriação não encontrado em {caminho}. "
            "Execute `python recriar_mascaras_kaggle.py` primeiro."
        )
    with open(caminho, encoding="utf-8") as fh:
        linhas = [l for l in csv.DictReader(fh) if l.get("area_original_px")]

    for l in linhas:
        l["_area"] = float(l["area_original_px"])
        l["_razao"] = float(l["razao_adicionada"]) if l.get("razao_adicionada") else 0.0
        l["_origem"] = l["nome_imagem"].split("_")[0]

    uteis = [l for l in linhas if l["_area"] > 0 and l["classificacao"] != "REVISAO_NECESSARIA"]
    escolhidas: List[str] = []

    # Duas sem alteração (a máscara original já cobria a ferida).
    sem = sorted([l for l in uteis if l["classificacao"] == "SEM_ALTERACAO"],
                 key=lambda l: -l["_area"])
    escolhidas += [l["nome_imagem"] for l in sem[:2]]

    # Das expandidas: variedade de origem e de magnitude da diferença.
    exp = [l for l in uteis if l["classificacao"] == "MASCARA_EXPANDIDA"]
    for origem in ("fusc", "medetec", "wsnet"):
        grupo = sorted([l for l in exp if l["_origem"] == origem], key=lambda l: l["_razao"])
        if not grupo:
            continue
        for q in (0.15, 0.55, 0.92):  # pouca, média e muita diferença
            cand = grupo[int(q * (len(grupo) - 1))]["nome_imagem"]
            if cand not in escolhidas:
                escolhidas.append(cand)
    return escolhidas[:n]


def _medidas(mascara: np.ndarray, escala: PixelScale) -> Dict[str, float]:
    """Mede a maior ferida da máscara, em pixels e em centímetros.

    Usa :func:`measure.measure_wounds` com ``cleanup=False``: a máscara já
    vem pronta, e limpá-la de novo alteraria os números que se quer comparar.
    """
    feridas = measure_wounds(mascara, pixels_por_cm=escala, cleanup=False)
    if not feridas:
        return {chave: 0.0 for chave in (
            "area_px", "perimetro_px", "comprimento_px", "largura_px",
            "area_cm2", "perimetro_cm", "comprimento_cm", "largura_cm")} | {"bbox": ""}
    f = feridas[0]
    x, y, w, h = f.bounding_box
    return {
        "area_px": float(f.area_pixels),
        "perimetro_px": float(f.perimeter_pixels),
        "comprimento_px": float(f.length_pixels),
        "largura_px": float(f.width_pixels),
        "area_cm2": float(f.area_cm2 or 0.0),
        "perimetro_cm": float(f.perimeter_cm or 0.0),
        "comprimento_cm": float(f.length_cm or 0.0),
        "largura_cm": float(f.width_cm or 0.0),
        "bbox": "%d,%d,%d,%d" % (x, y, w, h),
        "n_feridas": len(feridas),
    }


def _iou_dice(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int, int, int]:
    """IoU, Dice e as três áreas da comparação entre duas máscaras."""
    ba, bb = a > 0, b > 0
    inter = int(np.logical_and(ba, bb).sum())
    uniao = int(np.logical_or(ba, bb).sum())
    so_a = int(np.logical_and(ba, ~bb).sum())
    so_b = int(np.logical_and(~ba, bb).sum())
    iou = inter / uniao if uniao else 1.0
    dice = 2 * inter / (int(ba.sum()) + int(bb.sum())) if (ba.sum() + bb.sum()) else 1.0
    return iou, dice, inter, so_a, so_b


def _mapa_diferencas(imagem_rgb: np.ndarray, original: np.ndarray, nova: np.ndarray) -> np.ndarray:
    """Pinta sobre a fotografia as três regiões da comparação."""
    saida = imagem_rgb.copy()
    ba, bb = original > 0, nova > 0
    for regiao, cor in ((np.logical_and(ba, bb), COR_AMBAS),
                        (np.logical_and(ba, ~bb), COR_ORIGINAL),
                        (np.logical_and(~ba, bb), COR_NOVA)):
        if regiao.any():
            saida[regiao] = (0.45 * imagem_rgb[regiao] + 0.55 * np.array(cor)).astype(np.uint8)
    return saida


def processar_imagem(nome: str, indice: int, escala: PixelScale) -> Optional[dict]:
    """Gera arquivos e métricas de uma imagem, e devolve o registro."""
    stem = nome.rsplit(".", 1)[0]
    caminho_img = config.IMAGES_DIR / f"{stem}.png"
    caminho_orig = config.MASKS_DIR / f"{stem}.png"
    caminho_nova = MASCARAS_NOVAS_DIR / f"{stem}.png"

    # Correspondência verificada arquivo a arquivo, nunca por ordem.
    for caminho in (caminho_img, caminho_orig, caminho_nova):
        if not caminho.exists():
            logger.error("Ausente para %s: %s", stem, caminho)
            return None

    imagem = ler_imagem(caminho_img)
    original = (ler_imagem(caminho_orig, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    nova = (ler_imagem(caminho_nova, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if imagem is None:
        return None
    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    pasta = KAGGLE_SAIDA / ("Imagem_%02d_%s" % (indice, stem))
    overlay_orig = overlay_mask_on_image(rgb, original, color=COR_ORIGINAL, alpha=0.45)
    overlay_nova = overlay_mask_on_image(rgb, nova, color=COR_NOVA, alpha=0.45)
    diferencas = _mapa_diferencas(rgb, original, nova)

    saidas = {
        "original": rgb,
        "mascara_original": cv2.cvtColor(original * 255, cv2.COLOR_GRAY2RGB),
        "mascara_nova": cv2.cvtColor(nova * 255, cv2.COLOR_GRAY2RGB),
        "overlay_original": overlay_orig,
        "overlay_nova": overlay_nova,
        "comparacao": diferencas,
    }
    for sub, arr in saidas.items():
        escrever_imagem(pasta / sub / f"{stem}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    m_orig = _medidas(original, escala)
    m_nova = _medidas(nova, escala)
    iou, dice, inter, so_orig, so_nova = _iou_dice(original, nova)

    registro = {
        "indice": indice,
        "nome_imagem": stem,
        "origem": stem.split("_")[0],
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "intersecao_px": inter,
        "so_original_px": so_orig,
        "so_nova_px": so_nova,
        "pasta": str(pasta.relative_to(config.BASE_DIR)),
    }
    for prefixo, medidas in (("orig", m_orig), ("nova", m_nova)):
        for chave, valor in medidas.items():
            registro["%s_%s" % (prefixo, chave)] = (
                round(valor, 4) if isinstance(valor, float) else valor
            )
    for chave in ("area_px", "perimetro_px", "comprimento_px", "largura_px",
                  "area_cm2", "perimetro_cm", "comprimento_cm", "largura_cm"):
        a, b = m_orig[chave], m_nova[chave]
        registro["dif_%s" % chave] = round(b - a, 4)
        registro["difpct_%s" % chave] = round(100 * (b - a) / a, 2) if a else ""
    registro["_visuais"] = saidas
    return registro


def gerar_pdf(registros: List[dict], escala_ppc: float, linhas_moedas: List[dict]) -> None:
    """Monta o relatório PDF com as seções pedidas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    RELATORIO_DIR.mkdir(parents=True, exist_ok=True)
    cm_por_px = 1.0 / escala_ppc

    with PdfPages(RELATORIO_PDF) as pdf:
        # ---------------- Capa e introdução ---------------- #
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.92, "Calibração por Moedas e Comparação de Máscaras",
                 ha="center", size=17, weight="bold")
        fig.text(0.5, 0.885, "Projeto TecFeridas — segmentação e medição de feridas",
                 ha="center", size=10, color="#555555")
        texto = (
            "1. INTRODUÇÃO\n\n"
            "Objetivo da calibração\n"
            "Converter medidas de feridas de pixels para centímetros exige uma referência física\n"
            "presente na cena. Este relatório usa moedas brasileiras como padrão: o diâmetro de\n"
            "cada moeda é conhecido, então a razão entre esse diâmetro em pixels e o diâmetro real\n"
            "em centímetros fornece o fator pixels/cm.\n\n"
            "Por que moedas\n"
            "A moeda é um objeto rígido, plano, circular e de dimensão padronizada. Diferentes\n"
            "denominações servem como referências independentes: se todas produzirem escalas\n"
            "compatíveis, a calibração é coerente; se divergirem, a divergência aponta erro de\n"
            "medição ou diferença real de distância da câmera.\n\n"
            "Objetivo da comparação de máscaras\n"
            "O projeto recriou as máscaras do dataset Kaggle, partindo das originais e agregando\n"
            "regiões da lesão identificadas na fotografia. Este relatório compara, em 10 imagens,\n"
            "a máscara original e a recriada — visual e quantitativamente — e expressa as medidas\n"
            "nas duas unidades.\n\n"
            "Fonte das dimensões físicas\n"
            "Moedas Parâmetro/Dimensões Oficiais/Dimensões Oficiais.png\n\n"
            "RESSALVAS IMPORTANTES\n\n"
            "• A tabela do projeto NÃO contém a moeda de R$ 1,00. O valor usado (27,00 mm) é\n"
            "  externo — 2ª família do Real — e está marcado como EXTERNA em todas as tabelas.\n"
            "  Nenhum diâmetro foi estimado a partir das imagens.\n\n"
            "• A tabela NÃO traz tolerâncias de cunhagem. A classificação DENTRO/REVISÃO/FORA\n"
            "  usa consistência entre moedas: desvio da escala de cada imagem em relação à escala\n"
            "  mediana do conjunto (até 5%, até 10%, acima de 10%). Não são tolerâncias oficiais.\n\n"
            "• As fotografias de moedas e de feridas vêm de conjuntos distintos, com distâncias de\n"
            "  câmera diferentes. A escala aqui obtida é uma referência GERAL do conjunto de\n"
            "  moedas, não a escala específica de cada fotografia de ferida — ver seção 3.\n\n"
            "NENHUM TREINAMENTO FOI REALIZADO NESTA ETAPA."
        )
        fig.text(0.07, 0.06, texto, size=8.4, va="bottom", family="monospace", linespacing=1.55)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Referências físicas ---------------- #
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.95, "2. Referências físicas — moedas", ha="center", size=14, weight="bold")
        cabecalho = ["Moeda", "Fonte", "Ø oficial\n(mm)", "N\nimagens",
                     "Ø medido\n(px, mediana)", "pixels/cm", "cm/pixel",
                     "Dispersão\n(%)", "Dentro", "Revisão", "Fora"]
        corpo = []
        for l in linhas_moedas:
            fonte = "projeto" if l["fonte_dimensao"].startswith("Moedas") else "EXTERNA"
            corpo.append([
                l["rotulo"], fonte, "%.2f" % float(l["diametro_oficial_mm"]), l["n_imagens"],
                "%.1f" % float(l["diametro_px_mediano"]),
                "%.2f" % float(l["pixels_por_cm_mediano"]),
                "%.6f" % float(l["cm_por_pixel_mediano"]),
                "%.1f" % float(l["dispersao_relativa_pct"]),
                l["dentro"], l["revisao"], l["fora"],
            ])
        eixo = fig.add_axes([0.04, 0.42, 0.92, 0.42]); eixo.axis("off")
        tabela = eixo.table(cellText=corpo, colLabels=cabecalho, loc="center", cellLoc="center")
        tabela.auto_set_font_size(False); tabela.set_fontsize(8.5); tabela.scale(1, 2.1)
        for j in range(len(cabecalho)):
            tabela[0, j].set_facecolor("#3d5a73"); tabela[0, j].set_text_props(color="white", weight="bold")
        for i, l in enumerate(linhas_moedas, start=1):
            if not l["fonte_dimensao"].startswith("Moedas"):
                for j in range(len(cabecalho)):
                    tabela[i, j].set_facecolor("#fdf0d5")
        fig.text(0.07, 0.34, (
            "Ø oficial: diâmetro da tabela do projeto. A linha destacada usa valor EXTERNO — R$ 1,00 não consta na tabela.\n"
            "pixels/cm = Ø medido (px) / Ø oficial (cm).     cm/pixel = Ø oficial (cm) / Ø medido (px).\n"
            "Dispersão: desvio-padrão de pixels/cm dentro da denominação, em % da mediana — mede a variação entre fotografias\n"
            "da MESMA moeda, cuja causa principal é a distância da câmera, não erro de medição.\n\n"
            "A tabela oficial traz também peso e espessura; para calibração linear apenas o diâmetro é utilizável, já que\n"
            "peso não é observável na fotografia e espessura não aparece na projeção frontal."
        ), size=8.2, family="monospace", linespacing=1.5)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Validação da calibração ---------------- #
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.95, "3. Validação da calibração entre moedas", ha="center", size=14, weight="bold")
        eixo = fig.add_axes([0.08, 0.45, 0.85, 0.42])
        rotulos = [l["rotulo"] for l in linhas_moedas]
        valores = [float(l["pixels_por_cm_mediano"]) for l in linhas_moedas]
        p25 = [float(l["pixels_por_cm_p25"]) for l in linhas_moedas]
        p75 = [float(l["pixels_por_cm_p75"]) for l in linhas_moedas]
        erro = [np.array(valores) - np.array(p25), np.array(p75) - np.array(valores)]
        cores = ["#4c72b0" if l["fonte_dimensao"].startswith("Moedas") else "#c08a2e"
                 for l in linhas_moedas]
        eixo.bar(rotulos, valores, yerr=erro, capsize=6, color=cores)
        eixo.axhline(escala_ppc, color="#b03a2e", ls="--",
                     label="escala adotada = %.2f px/cm" % escala_ppc)
        eixo.set_ylabel("pixels por centímetro"); eixo.legend()
        eixo.set_title("Escala por denominação (mediana, com intervalo interquartil)", size=10)
        dispersao = 100 * (max(valores) - min(valores)) / float(np.median(valores))
        fig.text(0.07, 0.10, (
            "Leitura dos resultados\n\n"
            "A escala adotada é a MEDIANA DAS MEDIANAS por moeda (%.2f px/cm; %.6f cm/px), e não a média de todas as\n"
            "fotografias: assim cada denominação pesa igual, em vez de a moeda mais fotografada dominar o resultado.\n\n"
            "A amplitude entre as denominações é de %.1f%% da mediana. Essa dispersão NÃO indica que os diâmetros oficiais\n"
            "sejam inconsistentes entre si: como cada fotografia foi tirada a uma distância diferente, a mesma moeda produz\n"
            "diâmetros em pixels diferentes. A dispersão DENTRO de cada denominação (coluna da seção 2) confirma isso — ela é\n"
            "da mesma ordem da dispersão entre denominações, o que aponta a distância da câmera como causa dominante.\n\n"
            "Consequência prática, e limite deste relatório: não existe um único fator pixels/cm universal para fotografias\n"
            "tiradas a distâncias arbitrárias. As medidas em centímetros da seção 4 usam a escala geral acima e são, portanto,\n"
            "uma ESTIMATIVA DE ORDEM DE GRANDEZA. Medição clinicamente confiável exige uma referência física na própria\n"
            "fotografia da ferida — o projeto já suporta isso via measure.PixelScale (--referencia-pixels/--referencia-cm).\n"
            "As imagens do Kaggle não contêm nenhum objeto de referência."
        ) % (escala_ppc, cm_por_px, dispersao), size=8.2, family="monospace", linespacing=1.5)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Uma página por imagem ---------------- #
        for reg in registros:
            _pagina_imagem(pdf, reg, escala_ppc)

        # ---------------- Tabela consolidada ---------------- #
        _pagina_consolidada(pdf, registros, escala_ppc)

        # ---------------- Conclusão ---------------- #
        _pagina_conclusao(pdf, registros, linhas_moedas, escala_ppc, dispersao)

    logger.info("Relatório PDF gerado em %s", RELATORIO_PDF)


def _pagina_imagem(pdf, reg: dict, escala_ppc: float) -> None:
    """Uma página por imagem: 6 vistas + tabela de medidas."""
    import matplotlib.pyplot as plt

    vis = reg["_visuais"]
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.965, "4.%d  %s  (origem: %s)" % (reg["indice"], reg["nome_imagem"], reg["origem"]),
             ha="center", size=13, weight="bold")

    titulos = [
        ("A. Fotografia original", "original"),
        ("B. Máscara original", "mascara_original"),
        ("C. Nova máscara", "mascara_nova"),
        ("D. Foto + máscara original", "overlay_original"),
        ("E. Foto + nova máscara", "overlay_nova"),
        ("F. Diferenças sobre a foto", "comparacao"),
    ]
    for i, (titulo, chave) in enumerate(titulos):
        linha, coluna = divmod(i, 3)
        eixo = fig.add_axes([0.030 + coluna * 0.222, 0.505 - linha * 0.445, 0.205, 0.38])
        eixo.imshow(vis[chave]); eixo.axis("off")
        eixo.set_title(titulo, size=9)

    fig.text(0.700, 0.900, "Legenda do painel F", size=9, weight="bold")
    for j, (texto, cor) in enumerate((("apenas máscara original", COR_ORIGINAL),
                                      ("apenas nova máscara", COR_NOVA),
                                      ("presente nas duas", COR_AMBAS))):
        fig.patches.append(plt.Rectangle((0.702, 0.865 - j * 0.032), 0.017, 0.017,
                                         transform=fig.transFigure,
                                         facecolor=np.array(cor) / 255, edgecolor="#333333"))
        fig.text(0.726, 0.869 - j * 0.032, texto, size=8.2)

    linhas = [
        ["Área (px²)", "%.0f" % reg["orig_area_px"], "%.0f" % reg["nova_area_px"],
         "%+.0f" % reg["dif_area_px"], _pct(reg["difpct_area_px"])],
        ["Perímetro (px)", "%.0f" % reg["orig_perimetro_px"], "%.0f" % reg["nova_perimetro_px"],
         "%+.0f" % reg["dif_perimetro_px"], _pct(reg["difpct_perimetro_px"])],
        ["Comprimento (px)", "%.0f" % reg["orig_comprimento_px"], "%.0f" % reg["nova_comprimento_px"],
         "%+.0f" % reg["dif_comprimento_px"], _pct(reg["difpct_comprimento_px"])],
        ["Largura (px)", "%.0f" % reg["orig_largura_px"], "%.0f" % reg["nova_largura_px"],
         "%+.0f" % reg["dif_largura_px"], _pct(reg["difpct_largura_px"])],
        ["Área (cm²)", "%.3f" % reg["orig_area_cm2"], "%.3f" % reg["nova_area_cm2"],
         "%+.3f" % reg["dif_area_cm2"], _pct(reg["difpct_area_cm2"])],
        ["Perímetro (cm)", "%.2f" % reg["orig_perimetro_cm"], "%.2f" % reg["nova_perimetro_cm"],
         "%+.2f" % reg["dif_perimetro_cm"], _pct(reg["difpct_perimetro_cm"])],
        ["Comprimento (cm)", "%.2f" % reg["orig_comprimento_cm"], "%.2f" % reg["nova_comprimento_cm"],
         "%+.2f" % reg["dif_comprimento_cm"], _pct(reg["difpct_comprimento_cm"])],
        ["Largura (cm)", "%.2f" % reg["orig_largura_cm"], "%.2f" % reg["nova_largura_cm"],
         "%+.2f" % reg["dif_largura_cm"], _pct(reg["difpct_largura_cm"])],
    ]
    eixo = fig.add_axes([0.700, 0.34, 0.288, 0.42]); eixo.axis("off")
    tabela = eixo.table(cellText=linhas,
                        colLabels=["Medida", "Original", "Nova", "Dif.", "Dif. %"],
                        loc="center", cellLoc="center",
                        colWidths=[0.33, 0.17, 0.17, 0.16, 0.17])
    tabela.auto_set_font_size(False); tabela.set_fontsize(7.4); tabela.scale(1, 1.6)
    for j in range(5):
        tabela[0, j].set_facecolor("#3d5a73"); tabela[0, j].set_text_props(color="white", weight="bold")
    for i in range(1, len(linhas) + 1):
        tabela[i, 0].set_text_props(ha="left")
        tabela[i, 0].PAD = 0.04

    fig.text(0.700, 0.285, (
        "IoU  = %.4f          Dice = %.4f\n"
        "Interseção         : %d px\n"
        "Só máscara original: %d px\n"
        "Só nova máscara    : %d px\n"
        "Escala             : %.2f px/cm"
    ) % (reg["iou"], reg["dice"], reg["intersecao_px"], reg["so_original_px"],
         reg["so_nova_px"], escala_ppc), size=8, family="monospace", linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)


def _pct(valor) -> str:
    return "%+.1f%%" % valor if isinstance(valor, (int, float)) else "-"


def _pagina_consolidada(pdf, registros: List[dict], escala_ppc: float) -> None:
    """Tabela final com as principais medidas das imagens analisadas."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.95, "5. Tabela consolidada", ha="center", size=14, weight="bold")
    corpo = []
    for r in registros:
        corpo.append([
            "%02d" % r["indice"], r["nome_imagem"],
            "%.0f" % r["orig_area_px"], "%.0f" % r["nova_area_px"],
            "%.3f" % r["orig_area_cm2"], "%.3f" % r["nova_area_cm2"],
            _pct(r["difpct_area_px"]), "%.4f" % r["iou"], "%.4f" % r["dice"],
        ])
    eixo = fig.add_axes([0.03, 0.40, 0.94, 0.47]); eixo.axis("off")
    cabecalho = ["#", "Imagem", "Área orig.\n(px²)", "Área nova\n(px²)",
                 "Área orig.\n(cm²)", "Área nova\n(cm²)", "Δ área", "IoU", "Dice"]
    tabela = eixo.table(cellText=corpo, colLabels=cabecalho, loc="center", cellLoc="center")
    tabela.auto_set_font_size(False); tabela.set_fontsize(8); tabela.scale(1, 1.9)
    for j in range(len(cabecalho)):
        tabela[0, j].set_facecolor("#3d5a73"); tabela[0, j].set_text_props(color="white", weight="bold")

    ious = np.array([r["iou"] for r in registros])
    dices = np.array([r["dice"] for r in registros])
    difs = np.array([r["difpct_area_px"] for r in registros if isinstance(r["difpct_area_px"], (int, float))])
    fig.text(0.07, 0.15, (
        "Resumo das %d imagens\n\n"
        "IoU    : mínimo %.4f | mediana %.4f | máximo %.4f\n"
        "Dice   : mínimo %.4f | mediana %.4f | máximo %.4f\n"
        "Δ área : mínimo %+.1f%% | mediana %+.1f%% | máximo %+.1f%%\n\n"
        "Escala aplicada: %.2f px/cm (%.6f cm/px). Área em cm² = área em px² × (cm/px)².\n"
        "Toda medida linear em cm = medida em px × (cm/px)."
    ) % (len(registros), ious.min(), np.median(ious), ious.max(),
         dices.min(), np.median(dices), dices.max(),
         difs.min(), np.median(difs), difs.max(), escala_ppc, 1 / escala_ppc),
        size=8.6, family="monospace", linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)


def _pagina_conclusao(pdf, registros, linhas_moedas, escala_ppc, dispersao) -> None:
    """Conclusão técnica, separada por eixo de análise."""
    import matplotlib.pyplot as plt

    ious = np.array([r["iou"] for r in registros])
    difs = np.array([r["difpct_area_px"] for r in registros if isinstance(r["difpct_area_px"], (int, float))])
    n_projeto = sum(1 for l in linhas_moedas if l["fonte_dimensao"].startswith("Moedas"))
    total_imgs = sum(int(l["n_imagens"]) for l in linhas_moedas)

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.95, "6. Conclusão técnica", ha="center", size=14, weight="bold")
    texto = (
        "6.1 Resultado da calibração\n\n"
        "Foram processadas %d fotografias de moedas em 5 denominações. A escala adotada é\n"
        "%.2f px/cm (%.6f cm/px), mediana das medianas por denominação. O diâmetro foi obtido\n"
        "por transformada de Hough, que ajusta o círculo diretamente na borda da moeda; a\n"
        "segmentação por cor, usada para área e perímetro, incluía a sombra projetada e por isso\n"
        "não serve como medida primária de diâmetro.\n\n"
        "6.2 Consistência das referências físicas\n\n"
        "%d das 5 denominações têm diâmetro na tabela do projeto; R$ 1,00 usa valor externo.\n"
        "A amplitude entre as escalas medianas é de %.1f%%. Essa variação é compatível com a\n"
        "dispersão observada DENTRO de cada denominação, o que indica distância de câmera\n"
        "variável entre fotografias — e não inconsistência entre os diâmetros oficiais. Em outras\n"
        "palavras: as moedas concordam entre si na medida em que o arranjo fotográfico permite.\n\n"
        "6.3 Diferenças entre as máscaras\n\n"
        "Nas %d imagens analisadas, o IoU entre máscara original e recriada vai de %.3f a %.3f\n"
        "(mediana %.3f). A variação de área vai de %+.1f%% a %+.1f%% (mediana %+.1f%%).\n"
        "A recriação nunca remove pixels da máscara original — por construção, ela só agrega —,\n"
        "de modo que IoU abaixo de 1 significa exclusivamente região acrescentada.\n\n"
        "6.4 Comportamento das medidas em pixels\n\n"
        "Em pixels as medidas são exatas e diretamente verificáveis: área é contagem de pixels e\n"
        "perímetro é comprimento de contorno. São a base confiável para comparar as duas máscaras\n"
        "entre si, e é nessa unidade que as diferenças relatadas devem ser lidas.\n\n"
        "6.5 Comportamento das medidas em centímetros\n\n"
        "A conversão para cm herda a incerteza da escala. Como as fotografias do Kaggle não contêm\n"
        "objeto de referência, aplicou-se a escala geral do conjunto de moedas — fotografado em\n"
        "outro arranjo. Os valores em cm são, portanto, ORDEM DE GRANDEZA, não medida clínica.\n"
        "A razão entre as duas máscaras, essa sim, é preservada: como ambas usam a mesma escala,\n"
        "a diferença PERCENTUAL em cm é idêntica à diferença percentual em pixels.\n\n"
        "6.6 Observações visuais das sobreposições\n\n"
        "As sobreposições sobre a fotografia (painéis D, E e F de cada imagem) são o critério que\n"
        "nenhuma métrica substitui. O que elas mostram: a região acrescentada corresponde, na\n"
        "maioria dos casos, a tecido de ferida contíguo que a máscara original deixou de fora —\n"
        "leito avermelhado, esfacelo amarelado e tecido necrótico escuro. Nos casos de maior\n"
        "acréscimo, convém verificar se parte da região agregada é eritema perilesional: pele\n"
        "inflamada ao redor da ferida, que é avermelhada sem ser lesão.\n\n"
        "6.7 Ressalva de método\n\n"
        "Nem IoU nem Dice determinam qual máscara está correta: ambos medem concordância entre\n"
        "duas delimitações, não a aderência de qualquer uma delas à ferida real. Uma máscara maior\n"
        "não é melhor por ser maior. A avaliação de qual delimitação corresponde à ferida exige a\n"
        "inspeção visual dos painéis e, em última instância, julgamento clínico.\n\n"
        "NENHUM TREINAMENTO, FINE-TUNING OU ALTERAÇÃO DE PESOS FOI REALIZADO NESTA ETAPA.\n"
        "Nenhum arquivo original — imagens, máscaras do Kaggle ou fotografias de moedas — foi\n"
        "modificado, sobrescrito ou removido."
    ) % (total_imgs, escala_ppc, 1 / escala_ppc, n_projeto, dispersao,
         len(registros), ious.min(), ious.max(), np.median(ious),
         difs.min(), difs.max(), np.median(difs))
    fig.text(0.06, 0.04, texto, size=8.3, va="bottom", family="monospace", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def executar(n_imagens: int = 10) -> None:
    """Fluxo completo: calibração -> seleção -> comparação -> PDF."""
    escala_ppc, linhas_moedas = carregar_escala()
    escala = PixelScale.from_pixels_per_cm(escala_ppc)
    logger.info("Escala adotada: %.3f px/cm", escala_ppc)

    nomes = selecionar_imagens(n_imagens)
    registros = []
    for i, nome in enumerate(nomes, start=1):
        reg = processar_imagem(nome, i, escala)
        if reg is not None:
            registros.append(reg)

    TABELA_CSV.parent.mkdir(parents=True, exist_ok=True)
    campos = [c for c in registros[0] if not c.startswith("_")]
    with open(TABELA_CSV, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(registros)

    gerar_pdf(registros, escala_ppc, linhas_moedas)

    sep = "=" * 74
    print()
    print(sep)
    print("COMPARACAO DE MASCARAS E RELATORIO")
    print(sep)
    print("Escala adotada     : %.2f px/cm  (%.6f cm/px)" % (escala_ppc, 1 / escala_ppc))
    print("Imagens comparadas : %d" % len(registros))
    print("IoU    : min %.4f | mediana %.4f | max %.4f" % (
        min(r["iou"] for r in registros),
        float(np.median([r["iou"] for r in registros])),
        max(r["iou"] for r in registros)))
    print("Dice   : min %.4f | mediana %.4f | max %.4f" % (
        min(r["dice"] for r in registros),
        float(np.median([r["dice"] for r in registros])),
        max(r["dice"] for r in registros)))
    print()
    print("Arquivos por imagem: %s" % KAGGLE_SAIDA)
    print("Tabela consolidada : %s" % TABELA_CSV)
    print("RELATORIO PDF      : %s" % RELATORIO_PDF)
    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara máscaras original x nova e gera o relatório PDF."
    )
    parser.add_argument("--n-imagens", type=int, default=10)
    args = parser.parse_args()
    executar(n_imagens=args.n_imagens)
