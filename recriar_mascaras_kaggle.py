"""
recriar_mascaras_kaggle.py
===========================
Recria as máscaras do dataset Kaggle partindo das originais.

Para cada par (fotografia, máscara original), o script procura regiões da
lesão que tenham ficado de fora da máscara, analisando os PIXELS DA
FOTOGRAFIA numa vizinhança da máscara. O resultado é
``máscara original ∪ regiões aceitas``: a original nunca é reduzida, nem
sobrescrita — a saída vai para ``TecCurar_NewMasks``.

Área de investigação
--------------------
A partir da bounding box ``(x, y, w, h)`` da máscara original, com centro
``(cx, cy)``::

    nova largura = FATOR_EXPANSAO * w
    nova altura  = FATOR_EXPANSAO * h
    ROI          = centrada em (cx, cy), recortada nos limites da imagem

Com ``FATOR_EXPANSAO = 1.3``, cada lado cresce 1,3x. O fator é geométrico:
não é 1,3 pixel, nem limiar de cor, e estar dentro da ROI não inclui pixel
nenhum — a ROI é apenas onde se procura.

Decisão por três perfis
-----------------------
Em vez de perguntar "este pixel é avermelhado?", o script pergunta "este
pixel se parece mais com a FERIDA, com a PELE ou com o FUNDO desta
fotografia?". Os três perfis são construídos por imagem:

* **ferida** — modos de cor do interior da máscara original. A própria
  máscara é a melhor amostra disponível da lesão naquela foto. São vários
  modos, e não uma média: uma ferida costuma ter leito avermelhado,
  esfacelo amarelado e tecido escuro ao mesmo tempo, e a média desses três
  não se parece com nenhum deles.
* **pele** — modos de um anel em volta da lesão, pulando a borda imediata.
* **fundo** — modos de uma faixa na borda da imagem, fora da ROI. Cobre o
  caso de um objeto, lençol ou superfície ter cor parecida com ferida.

Um candidato só entra se for mais próximo da ferida do que da pele E do
fundo. Roxo, vermelho, rosa e branco recebem atenção especial na forma de
uma margem mais permissiva — nunca de inclusão automática.

Uso:
    python recriar_mascaras_kaggle.py --limite 60 --amostras 6   # prototipo
    python recriar_mascaras_kaggle.py                            # dataset todo
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from utils import escrever_imagem, get_logger, ler_imagem, overlay_mask_on_image

logger = get_logger(__name__, config.LOGS_DIR / "recriar_mascaras_kaggle.log")

SAIDA_PADRAO = config.BASE_DIR / "TecCurar_NewMasks"
RELATORIO = config.REPORTS_DIR / "recriacao_kaggle.csv"
AMOSTRAS_DIR = config.FIGURES_DIR / "recriacao_kaggle"

# --------------------------------------------------------------------------- #
# Parâmetros
# --------------------------------------------------------------------------- #

# Fator GEOMÉTRICO de expansão do bounding box (cada lado). Ver docstring.
FATOR_EXPANSAO = 1.3

# Quantos modos de cor representam cada perfil.
MODOS_FERIDA = 3
# Três modos de pele, e não dois: em membro inflamado a pele perilesional é
# avermelhada e precisa de um modo próprio. Com dois, os centros caem na
# pele pálida e todo o eritema passa a parecer "mais ferida que pele".
MODOS_PELE = 3
MODOS_FUNDO = 2

# Buraco interno acima desta fração da máscara original não é falha de
# segmentação: é pele entre lesões, e não deve ser fechado.
FRACAO_MAXIMA_BURACO = 0.15

# Um candidato precisa estar mais perto da ferida que da pele e do fundo:
#   d_ferida <= margem * min(d_pele, d_fundo)
# Abaixo de 1 a exigência é de folga, não de empate. As cores de atenção
# especial usam a margem frouxa — continuam precisando vencer pele e fundo,
# mas sem a folga extra.
MARGEM_PADRAO = 0.80
MARGEM_COR_ALVO = 0.92

# Piso de diferença em relação à pele, em ΔE CIE76. Impede que um pixel
# praticamente da cor da pele entre só por estar ligeiramente mais perto
# de algum modo da ferida.
DELTA_PELE_MINIMO = 14.0

# Região candidata mínima: fração da máscara original e piso absoluto.
# Pixel solto é ruído de compressão, reflexo ou sombra — nunca lesão.
FRACAO_MINIMA_COMPONENTE = 0.02
AREA_MINIMA_COMPONENTE_PX = 40

# Continuidade espacial: distância máxima entre o componente e a máscara.
DISTANCIA_MAXIMA_PX = 12

# Anel de pele: entre estas duas dilatações da máscara.
ANEL_INTERNO_PX = 6
ANEL_EXTERNO_PX = 60

# Faixa da borda da imagem usada como amostra de fundo.
FAIXA_FUNDO = 0.08

MIN_PIXELS_PERFIL = 200

# Acima desta razão (adicionado / original) o resultado vai para revisão.
#
# 0,75 vem da inspeção visual: acima disso o acréscimo deixa de parecer
# borda de ferida que escapou da máscara e passa a cobrir área comparável à
# lesão inteira. Em membro muito eritematoso a pele perilesional tem cor
# próxima do leito, e nenhum limiar de cor os separa com segurança — o
# honesto é marcar o caso para revisão em vez de afirmar que está correto.
RAZAO_SUSPEITA = 0.75

SEM_ALTERACAO = "SEM_ALTERACAO"
MASCARA_EXPANDIDA = "MASCARA_EXPANDIDA"
REVISAO_NECESSARIA = "REVISAO_NECESSARIA"


# --------------------------------------------------------------------------- #
# Perfis de cor
# --------------------------------------------------------------------------- #

def modos_de_cor(pixels: np.ndarray, k: int) -> Optional[np.ndarray]:
    """Resume um conjunto de pixels Lab em ``k`` cores representativas.

    Usa k-means porque a média de uma região heterogênea não representa
    nenhuma de suas partes: a média entre leito vermelho e esfacelo branco
    é um rosa que não existe na ferida.

    Returns:
        Matriz ``(k', 3)`` com os centros, ou ``None`` se houver poucos
        pixels para uma estimativa confiável.
    """
    if len(pixels) < MIN_PIXELS_PERFIL:
        return None
    amostra = pixels.astype(np.float32)
    if len(amostra) > 20000:  # k-means não precisa da região inteira
        idx = np.random.default_rng(config.SEED).choice(len(amostra), 20000, replace=False)
        amostra = amostra[idx]
    k_efetivo = min(k, max(1, len(amostra) // MIN_PIXELS_PERFIL))
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centros = cv2.kmeans(amostra, k_efetivo, None, criterio, 3, cv2.KMEANS_PP_CENTERS)
    return centros


def distancia_aos_modos(lab: np.ndarray, modos: np.ndarray) -> np.ndarray:
    """Distância CIE76 de cada pixel ao modo de cor MAIS PRÓXIMO."""
    distancias = [np.linalg.norm(lab - m.reshape(1, 1, 3), axis=2) for m in modos]
    return np.min(np.stack(distancias, axis=0), axis=0)


def cores_de_atencao(hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Pixels roxos, vermelhos, rosas ou brancos (H em 0-179, convenção OpenCV).

    O vermelho ocupa as duas pontas da roda de matiz, por isso duas faixas.
    O branco é definido por saturação baixa com luminosidade alta.

    Marcar um pixel aqui NÃO o inclui na máscara: apenas lhe concede a
    margem mais permissiva na comparação entre ferida, pele e fundo.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vermelho_rosa = ((h <= 12) | (h >= 160)) & (s >= 50) & (v >= 50)
    roxo = (h >= 125) & (h < 160) & (s >= 40) & (v >= 40)
    branco = (s <= 50) & (v >= 160)
    return vermelho_rosa | roxo | branco


# --------------------------------------------------------------------------- #
# Geometria
# --------------------------------------------------------------------------- #

def binarizar(mascara: np.ndarray) -> np.ndarray:
    """Converte para 0/1 com o mesmo limiar do ``WoundDataset`` (> 127)."""
    if mascara.ndim == 3:
        mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
    return (mascara > 127).astype(np.uint8)


def calcular_roi(mascara: np.ndarray, fator: float = FATOR_EXPANSAO) -> Tuple[int, int, int, int]:
    """ROI expandida ``(x0, y0, x1, y1)``, recortada nos limites da imagem."""
    altura, largura = mascara.shape
    ys, xs = np.nonzero(mascara)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
    cx, cy = x + w / 2.0, y + h / 2.0
    novo_w, novo_h = w * fator, h * fator
    return (
        max(0, int(round(cx - novo_w / 2.0))),
        max(0, int(round(cy - novo_h / 2.0))),
        min(largura, int(round(cx + novo_w / 2.0))),
        min(altura, int(round(cy + novo_h / 2.0))),
    )


def _nucleo(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def preencher_buracos_pequenos(mascara: np.ndarray, area_referencia: int) -> np.ndarray:
    """Fecha apenas os buracos internos pequenos da máscara.

    Preencher todo contorno externo seria destrutivo numa imagem com várias
    feridas: assim que regiões candidatas fazem ponte entre duas lesões, o
    preenchimento engole toda a pele entre elas. Aqui cada buraco é medido,
    e só os que representam falha de segmentação — pequenos frente à lesão —
    são fechados.
    """
    contornos, hierarquia = cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarquia is None:
        return mascara
    limite = FRACAO_MAXIMA_BURACO * max(area_referencia, 1)
    saida = mascara.copy()
    for i, contorno in enumerate(contornos):
        if hierarquia[0][i][3] < 0:  # contorno externo, não é buraco
            continue
        if cv2.contourArea(contorno) <= limite:
            cv2.drawContours(saida, [contorno], -1, color=1, thickness=cv2.FILLED)
    return saida


# --------------------------------------------------------------------------- #
# Processamento
# --------------------------------------------------------------------------- #

def processar_par(caminho_imagem: Path, caminho_mascara: Path) -> Tuple[Optional[np.ndarray], dict]:
    """Analisa um par e devolve ``(nova_mascara_0_1, registro)``."""
    reg: Dict[str, object] = {"nome_imagem": caminho_imagem.name}

    imagem = ler_imagem(caminho_imagem)
    bruta = ler_imagem(caminho_mascara, cv2.IMREAD_GRAYSCALE)
    if imagem is None or bruta is None:
        reg.update(classificacao=REVISAO_NECESSARIA, motivo="falha ao ler imagem ou mascara")
        return None, reg

    if bruta.shape[:2] != imagem.shape[:2]:
        bruta = cv2.resize(bruta, (imagem.shape[1], imagem.shape[0]), interpolation=cv2.INTER_NEAREST)

    original = binarizar(bruta)
    area_original = int(original.sum())
    reg["area_original_px"] = area_original

    if area_original == 0:
        reg.update(classificacao=REVISAO_NECESSARIA, motivo="mascara original vazia",
                   area_nova_px=0, area_adicionada_px=0)
        return original, reg

    altura, largura = original.shape
    x0, y0, x1, y1 = calcular_roi(original)
    reg["roi"] = "%d,%d,%d,%d" % (x0, y0, x1, y1)

    roi = np.zeros_like(original)
    roi[y0:y1, x0:x1] = 1

    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # --- perfil da FERIDA: interior da máscara original ------------------- #
    interior = cv2.erode(original, _nucleo(2)) if area_original > 400 else original
    modos_ferida = modos_de_cor(lab[interior > 0], MODOS_FERIDA)
    if modos_ferida is None:
        modos_ferida = modos_de_cor(lab[original > 0], MODOS_FERIDA)
    if modos_ferida is None:
        reg.update(classificacao=REVISAO_NECESSARIA, motivo="lesao pequena demais para perfil de cor",
                   area_nova_px=area_original, area_adicionada_px=0)
        return original, reg

    # --- perfil da PELE: anel em volta da lesão, dentro da ROI ------------ #
    anel = ((cv2.dilate(original, _nucleo(ANEL_EXTERNO_PX)) > 0)
            & (cv2.dilate(original, _nucleo(ANEL_INTERNO_PX)) == 0)
            & (roi > 0)).astype(np.uint8)
    modos_pele = modos_de_cor(lab[anel > 0], MODOS_PELE)
    if modos_pele is None:
        reg.update(classificacao=REVISAO_NECESSARIA, motivo="pele insuficiente para referencia",
                   area_nova_px=area_original, area_adicionada_px=0)
        return original, reg

    # --- perfil do FUNDO: faixa da borda, fora da ROI --------------------- #
    borda = np.zeros_like(original)
    mb, nb = int(altura * FAIXA_FUNDO), int(largura * FAIXA_FUNDO)
    borda[:mb, :] = borda[-mb:, :] = borda[:, :nb] = borda[:, -nb:] = 1
    borda[roi > 0] = 0
    modos_fundo = modos_de_cor(lab[borda > 0], MODOS_FUNDO)
    reg["tem_perfil_fundo"] = modos_fundo is not None

    d_ferida = distancia_aos_modos(lab, modos_ferida)
    d_pele = distancia_aos_modos(lab, modos_pele)
    d_fundo = distancia_aos_modos(lab, modos_fundo) if modos_fundo is not None else np.full_like(d_pele, np.inf)
    d_rival = np.minimum(d_pele, d_fundo)

    # --- candidatos -------------------------------------------------------- #
    atencao = cores_de_atencao(hsv, lab)
    margem = np.where(atencao, MARGEM_COR_ALVO, MARGEM_PADRAO)

    limite_distancia = float(np.clip(0.25 * np.sqrt(area_original), 8, 60))
    distancia = cv2.distanceTransform((original == 0).astype(np.uint8), cv2.DIST_L2, 5)
    reg["limite_distancia_px"] = round(limite_distancia, 1)

    candidatos = (
        (roi > 0)
        & (original == 0)
        & (distancia <= limite_distancia)
        & (d_pele >= DELTA_PELE_MINIMO)
        & (d_ferida <= margem * d_rival)
    ).astype(np.uint8)

    # --- análise espacial: regiões, não pixels soltos ---------------------- #
    area_minima = max(AREA_MINIMA_COMPONENTE_PX, FRACAO_MINIMA_COMPONENTE * area_original)
    vizinhanca = cv2.dilate(original, _nucleo(DISTANCIA_MAXIMA_PX))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidatos, connectivity=8)

    aceitos = np.zeros_like(original)
    n_aceitos = descartados_area = descartados_longe = 0
    for rotulo in range(1, n_labels):
        if stats[rotulo, cv2.CC_STAT_AREA] < area_minima:
            descartados_area += 1
            continue
        componente = labels == rotulo
        if not np.any(componente & (vizinhanca > 0)):
            descartados_longe += 1
            continue
        aceitos[componente] = 1
        n_aceitos += 1

    nova = ((original | aceitos) > 0).astype(np.uint8)

    nova = preencher_buracos_pequenos(nova, area_original)
    # Garantia dura: a original nunca é reduzida.
    nova = ((nova | original) > 0).astype(np.uint8)

    area_nova = int(nova.sum())
    adicionada = area_nova - area_original
    reg.update(
        area_nova_px=area_nova,
        area_adicionada_px=adicionada,
        razao_adicionada=round(adicionada / area_original, 4),
        aumento_pct=round(100 * adicionada / area_original, 2),
        componentes_aceitos=n_aceitos,
        componentes_descartados_area=descartados_area,
        componentes_descartados_distancia=descartados_longe,
        modos_ferida=len(modos_ferida),
        modos_pele=len(modos_pele),
        modos_fundo=len(modos_fundo) if modos_fundo is not None else 0,
    )

    if aceitos.any():
        sel = aceitos > 0
        reg["dist_ferida_media"] = round(float(d_ferida[sel].mean()), 2)
        reg["dist_pele_media"] = round(float(d_pele[sel].mean()), 2)
        reg["dist_fundo_media"] = ("" if modos_fundo is None
                                   else round(float(d_fundo[sel].mean()), 2))
        reg["frac_cor_atencao"] = round(float(atencao[sel].mean()), 3)
        lab_add = lab[sel].mean(axis=0)
        reg["lab_medio_adicionado"] = "%.0f,%.0f,%.0f" % tuple(lab_add)
        ys, xs = np.nonzero(aceitos)
        reg["bbox_adicionado"] = "%d,%d,%d,%d" % (xs.min(), ys.min(),
                                                  xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)

    # O preenchimento de buracos pode aumentar a área sem que nenhuma região
    # candidata tenha sido aceita — são furos internos da máscara original
    # sendo fechados. É um acréscimo legítimo, mas de origem diferente, e
    # registrá-lo à parte evita atribuir à análise de cor um ganho que veio
    # só da topologia.
    reg["origem_acrescimo"] = ("nenhuma" if adicionada == 0
                               else "analise_de_cor" if aceitos.any()
                               else "preenchimento_de_buracos")

    if adicionada == 0:
        reg.update(classificacao=SEM_ALTERACAO, motivo="")
    elif adicionada / area_original > RAZAO_SUSPEITA:
        reg.update(classificacao=REVISAO_NECESSARIA, motivo="area adicionada desproporcional")
    else:
        reg.update(classificacao=MASCARA_EXPANDIDA,
                   motivo=("regiao mais proxima da ferida que da pele e do fundo"
                           if aceitos.any() else "buracos internos preenchidos"))
    return nova, reg


def salvar_amostra(imagem: np.ndarray, original: np.ndarray, nova: np.ndarray,
                   roi: Tuple[int, int, int, int], destino: Path) -> None:
    """Painel de conferência com 7 vistas lado a lado."""
    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    adicionado = ((nova > 0) & (original == 0)).astype(np.uint8)

    com_roi = rgb.copy()
    x0, y0, x1, y1 = roi
    cv2.rectangle(com_roi, (x0, y0), (x1 - 1, y1 - 1), (60, 130, 255), 2)

    paineis = [
        ("Imagem original", rgb),
        ("Mascara original", cv2.cvtColor(original * 255, cv2.COLOR_GRAY2RGB)),
        ("Mascara nova", cv2.cvtColor(nova * 255, cv2.COLOR_GRAY2RGB)),
        ("Imagem + original", overlay_mask_on_image(rgb, original, color=(220, 50, 50))),
        ("Imagem + nova", overlay_mask_on_image(rgb, nova, color=(40, 200, 90))),
        ("Area de analise 1,3x", com_roi),
        ("Regioes adicionadas", overlay_mask_on_image(rgb, adicionado, color=(250, 190, 40))),
    ]
    faixa = np.full((26, rgb.shape[1], 3), 255, dtype=np.uint8)
    colunas = []
    for titulo, painel in paineis:
        rotulo = faixa.copy()
        cv2.putText(rotulo, titulo, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        colunas.append(np.vstack([rotulo, painel[: rgb.shape[0]]]))
    escrever_imagem(destino, cv2.cvtColor(np.hstack(colunas), cv2.COLOR_RGB2BGR))


def recriar(images_dir: Optional[Path] = None, masks_dir: Optional[Path] = None,
            output_dir: Optional[Path] = None, limite: Optional[int] = None,
            amostras: int = 0) -> Dict[str, int]:
    """Processa o dataset e grava uma nova máscara para CADA imagem."""
    images_dir = images_dir or config.IMAGES_DIR
    masks_dir = masks_dir or config.MASKS_DIR
    output_dir = output_dir or SAIDA_PADRAO
    output_dir.mkdir(parents=True, exist_ok=True)

    extensoes = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    imagens = sorted(p for p in images_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in extensoes)
    if limite:
        imagens = imagens[:limite]
    mascaras = {p.stem.lower(): p for p in masks_dir.iterdir()
                if p.is_file() and p.suffix.lower() in extensoes}

    if amostras:
        AMOSTRAS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        iterador = tqdm(imagens, desc="recriando máscaras")
    except ImportError:  # pragma: no cover
        iterador = imagens

    linhas: List[dict] = []
    sem_mascara: List[str] = []
    contagem = {SEM_ALTERACAO: 0, MASCARA_EXPANDIDA: 0, REVISAO_NECESSARIA: 0}
    salvas = 0

    for caminho_imagem in iterador:
        caminho_mascara = mascaras.get(caminho_imagem.stem.lower())
        if caminho_mascara is None:
            sem_mascara.append(caminho_imagem.name)
            continue

        nova, reg = processar_par(caminho_imagem, caminho_mascara)
        contagem[reg["classificacao"]] = contagem.get(reg["classificacao"], 0) + 1

        if nova is None:
            # Ilegível: a original é reproduzida como está, para que toda
            # imagem tenha uma máscara na pasta nova.
            shutil.copy2(caminho_mascara, output_dir / f"{caminho_imagem.stem}.png")
        else:
            escrever_imagem(output_dir / f"{caminho_imagem.stem}.png", (nova * 255).astype(np.uint8))
        linhas.append(reg)

        if amostras and salvas < amostras and reg["classificacao"] == MASCARA_EXPANDIDA:
            imagem = ler_imagem(caminho_imagem)
            original = binarizar(ler_imagem(caminho_mascara, cv2.IMREAD_GRAYSCALE))
            if imagem is not None and nova is not None:
                x0, y0, x1, y1 = (int(v) for v in str(reg["roi"]).split(","))
                salvar_amostra(imagem, original, nova, (x0, y0, x1, y1),
                               AMOSTRAS_DIR / f"{caminho_imagem.stem}_comparacao.png")
                salvas += 1

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    campos = sorted({c for l in linhas for c in l})
    ordem = ["nome_imagem", "classificacao", "motivo"] + [c for c in campos if c not in
                                                          ("nome_imagem", "classificacao", "motivo")]
    with open(RELATORIO, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=ordem)
        escritor.writeheader()
        escritor.writerows(linhas)

    _relatar(imagens, linhas, contagem, sem_mascara, output_dir, salvas)
    return contagem


def _relatar(imagens, linhas, contagem, sem_mascara, output_dir, amostras) -> None:
    """Relatório de validação da recriação."""
    sep = "=" * 72
    expandidas = [l for l in linhas if l["classificacao"] == MASCARA_EXPANDIDA]
    razoes = np.array([l["razao_adicionada"] for l in expandidas]) if expandidas else np.array([])

    print()
    print(sep)
    print("RECRIACAO DE MASCARAS -- expansao %.1fx da bounding box" % FATOR_EXPANSAO)
    print(sep)
    print("Imagens encontradas          : %d" % len(imagens))
    print("Imagens sem mascara original : %d" % len(sem_mascara))
    print("Novas mascaras gravadas      : %d" % len(linhas))
    print()
    print("Classificacao:")
    for chave in (SEM_ALTERACAO, MASCARA_EXPANDIDA, REVISAO_NECESSARIA):
        n = contagem.get(chave, 0)
        print("  %-20s %6d  (%.1f%%)" % (chave, n, 100 * n / max(len(linhas), 1)))
    if razoes.size:
        print()
        print("Area adicionada nas expandidas (fracao da original):")
        print("  mediana=%.3f  media=%.3f  p90=%.3f  maximo=%.3f" % (
            np.median(razoes), razoes.mean(), np.percentile(razoes, 90), razoes.max()))
        atencao = [l["frac_cor_atencao"] for l in expandidas
                   if isinstance(l.get("frac_cor_atencao"), (int, float))]
        if atencao:
            print("  fracao dos pixels adicionados em cor de atencao: mediana %.2f" % np.median(atencao))
        print()
        print("Origem do acrescimo:")
        for origem in ("analise_de_cor", "preenchimento_de_buracos"):
            n = sum(1 for l in linhas if l.get("origem_acrescimo") == origem)
            print("  %-26s %6d" % (origem, n))
        com_fundo = sum(1 for l in linhas if l.get("tem_perfil_fundo"))
        print("  %-26s %6d de %d imagens" % ("perfil de fundo obtido", com_fundo, len(linhas)))
    motivos: Dict[str, int] = {}
    for l in linhas:
        if l["classificacao"] == REVISAO_NECESSARIA:
            motivos[l.get("motivo", "")] = motivos.get(l.get("motivo", ""), 0) + 1
    if motivos:
        print()
        print("Motivos de revisao:")
        for motivo, n in sorted(motivos.items(), key=lambda kv: -kv[1]):
            print("  %-45s %d" % (motivo, n))
    if sem_mascara:
        print()
        print("SEM MASCARA ORIGINAL (%d): %s" % (len(sem_mascara), ", ".join(sem_mascara[:8])))
    print()
    print("Novas mascaras : %s" % output_dir)
    print("Relatorio      : %s" % RELATORIO)
    if amostras:
        print("Amostras (%d)  : %s" % (amostras, AMOSTRAS_DIR))
    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recria as mascaras do Kaggle comparando ferida, pele e fundo."
    )
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--amostras", type=int, default=0)
    args = parser.parse_args()

    recriar(
        images_dir=Path(args.images_dir) if args.images_dir else None,
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limite=args.limite,
        amostras=args.amostras,
    )
