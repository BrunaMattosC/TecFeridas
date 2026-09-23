"""
recriar_mascaras_dataset.py
============================
Recria as máscaras do dataset usando as originais como ponto de partida.

Para cada par (imagem, máscara original), o script procura regiões da lesão
que tenham ficado de fora da máscara, analisando os PIXELS DA IMAGEM em uma
vizinhança da máscara, e devolve ``máscara original ∪ regiões aceitas``.

A máscara original nunca é reduzida, nem sobrescrita: a saída vai para uma
pasta separada.

Como a área de análise é calculada
-----------------------------------
A partir da bounding box ``(x, y, w, h)`` da máscara original, com centro
``(cx, cy)``::

    nova largura = FATOR_EXPANSAO * w
    nova altura  = FATOR_EXPANSAO * h
    ROI          = centrada em (cx, cy), recortada nos limites da imagem

Com ``FATOR_EXPANSAO = 1.5``, cada lado cresce 1,5x — a ÁREA da ROI fica
2,25x a do bbox, consequência de escalar os dois lados. O fator é
geométrico: não é 1,5 pixel, não é limiar de cor, e estar dentro da ROI
não inclui pixel nenhum. A ROI é apenas o campo de busca.

Critério de inclusão
--------------------
Um pixel fora da máscara só entra se, cumulativamente:

1. estiver dentro da ROI;
2. diferir da pele predominante daquela imagem acima de ``DELTA_E_MINIMO``
   (distância CIE76 no espaço Lab, perceptualmente uniforme);
3. pertencer a uma das famílias de cor de interesse — vermelho, rosa, roxo
   ou branco;
4. fazer parte de um componente conectado grande o bastante
   (``AREA_MINIMA_COMPONENTE``);
5. esse componente encostar ou quase encostar na máscara original
   (``DISTANCIA_MAXIMA_PX``).

A cor da pele é estimada IMAGE A IMAGEM, num anel em volta da lesão — nunca
com um valor fixo para todo o dataset.

Uso:
    python recriar_mascaras_dataset.py --limite 40 --amostras 6   # prototipo
    python recriar_mascaras_dataset.py                            # dataset todo
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
from utils import escrever_imagem, get_logger, ler_imagem

logger = get_logger(__name__, config.LOGS_DIR / "recriar_mascaras_dataset.log")

SAIDA_PADRAO = config.DATASET_DIR / "masks_recriadas"
RELATORIO = config.REPORTS_DIR / "recriacao_mascaras.csv"
AMOSTRAS_DIR = config.FIGURES_DIR / "recriacao_mascaras"

# --------------------------------------------------------------------------- #
# Parâmetros da análise
# --------------------------------------------------------------------------- #

# Fator GEOMÉTRICO de expansão do bounding box (cada lado). Ver docstring.
FATOR_EXPANSAO = 1.5

# Diferença mínima em relação à pele predominante, em ΔE CIE76 (Lab).
# ~2,3 é o limiar de percepção humana; 18 exige uma diferença bem visível,
# para não capturar variação normal de tom, sombra suave ou rubor.
DELTA_E_MINIMO = 18.0

# Quão mais perto da LESÃO que da PELE o pixel precisa estar, em ΔE, para
# ser aceito: exige ``delta_lesao <= MARGEM_LESAO * delta_pele``. Com 0,8 o
# pixel tem de ser claramente mais parecido com a lesão, e não apenas
# empatado — é o que separa tecido de ferida do eritema em volta dela.
MARGEM_LESAO = 0.8

# Um candidato precisa formar uma região com pelo menos esta fração da área
# da máscara original (e nunca menos que o piso absoluto em pixels).
FRACAO_MINIMA_COMPONENTE = 0.02
AREA_MINIMA_COMPONENTE_PX = 40

# Distância máxima, em pixels, entre o componente candidato e a máscara
# original. Impõe a continuidade espacial: cor parecida longe da lesão não
# entra (regra 11/12 do escopo).
DISTANCIA_MAXIMA_PX = 12

# Anel de pele usado como referência: entre estas duas dilatações da máscara.
# A borda imediata é pulada porque mistura lesão e pele.
ANEL_INTERNO_PX = 6
ANEL_EXTERNO_PX = 60

# Mínimo de pixels de pele para a estimativa ser considerada confiável.
MIN_PIXELS_PELE = 300

# Acima desta razão (área adicionada / área original) o resultado é tratado
# como suspeito e marcado para revisão, em vez de aceito em silêncio.
RAZAO_SUSPEITA = 1.5

SEM_ALTERACAO = "SEM_ALTERACAO"
MASCARA_EXPANDIDA = "MASCARA_EXPANDIDA"
REVISAO_NECESSARIA = "REVISAO_NECESSARIA"


def binarizar(mascara: np.ndarray) -> np.ndarray:
    """Converte a máscara para 0/1 usando o mesmo limiar do ``WoundDataset``."""
    if mascara.ndim == 3:
        mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
    return (mascara > 127).astype(np.uint8)


def calcular_roi(mascara_bin: np.ndarray, fator: float = FATOR_EXPANSAO) -> Tuple[int, int, int, int]:
    """Devolve a ROI expandida ``(x0, y0, x1, y1)`` a partir do bbox da máscara.

    Cada lado do bounding box é multiplicado por ``fator``, mantendo o
    centro, e o resultado é recortado nos limites da imagem — nunca
    devolve coordenada negativa nem maior que a resolução.
    """
    altura, largura = mascara_bin.shape
    ys, xs = np.nonzero(mascara_bin)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1

    cx, cy = x + w / 2.0, y + h / 2.0
    novo_w, novo_h = w * fator, h * fator

    x0 = max(0, int(round(cx - novo_w / 2.0)))
    y0 = max(0, int(round(cy - novo_h / 2.0)))
    x1 = min(largura, int(round(cx + novo_w / 2.0)))
    y1 = min(altura, int(round(cy + novo_h / 2.0)))
    return x0, y0, x1, y1


def estimar_pele(lab: np.ndarray, anel: np.ndarray) -> Optional[np.ndarray]:
    """Estima a cor predominante da pele num anel ao redor da lesão.

    Trabalha em Lab e busca o PICO do histograma 2-D de cromaticidade
    ``(a, b)`` — a cor que ocupa a maior área, e não a média, que seria
    puxada por sombra, fundo ou reflexo. Antes disso, descarta os extremos
    de luminância, onde ficam sombras e brilhos especulares.

    Returns:
        Vetor Lab (float32) da pele predominante, ou ``None`` se não houver
        pixels suficientes para uma estimativa confiável.
    """
    pixels = lab[anel > 0]
    if len(pixels) < MIN_PIXELS_PELE:
        return None

    luminancia = pixels[:, 0]
    baixo, alto = np.percentile(luminancia, [10, 90])
    plausiveis = pixels[(luminancia >= baixo) & (luminancia <= alto)]
    if len(plausiveis) < MIN_PIXELS_PELE:
        plausiveis = pixels

    # Pico do histograma de cromaticidade, em células de 4 unidades.
    a_idx = np.clip(plausiveis[:, 1].astype(np.int32) // 4, 0, 63)
    b_idx = np.clip(plausiveis[:, 2].astype(np.int32) // 4, 0, 63)
    hist = np.zeros((64, 64), dtype=np.int32)
    np.add.at(hist, (a_idx, b_idx), 1)
    pa, pb = np.unravel_index(int(np.argmax(hist)), hist.shape)

    # Refina com os pixels vizinhos do pico, para não ficar preso à célula.
    perto = (np.abs(a_idx - pa) <= 1) & (np.abs(b_idx - pb) <= 1)
    return np.median(plausiveis[perto], axis=0).astype(np.float32)


def cores_de_interesse(hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Máscara booleana dos pixels vermelhos, rosas, roxos ou brancos.

    Matiz em convenção OpenCV (H em 0-179). O vermelho ocupa as duas pontas
    da roda, por isso aparece em duas faixas. O branco é tratado por
    saturação baixa com luminosidade alta, e não por matiz.

    O branco exige ainda um viés AMARELADO (canal ``b`` do Lab acima do
    neutro): esfacelo e fibrina são brancos amarelados, enquanto gaze,
    lençol e fundo clínico são brancos neutros ou azulados. Sem essa
    condição, o fundo da foto entra na máscara.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b = lab[..., 2]  # em Lab do OpenCV, 128 é o neutro

    vermelho_rosa = (((h <= 12) | (h >= 160)) & (s >= 55) & (v >= 55))
    roxo = ((h >= 125) & (h < 160) & (s >= 45) & (v >= 45))
    branco = ((s <= 45) & (v >= 165) & (b >= 136))
    return vermelho_rosa | roxo | branco


def processar_par(
    caminho_imagem: Path, caminho_mascara: Path
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """Analisa um par imagem/máscara e devolve ``(nova_mascara, registro)``.

    ``nova_mascara`` é 0/1; ``registro`` traz a classificação e as métricas
    que alimentam o relatório e a validação.
    """
    registro: Dict[str, object] = {"nome_imagem": caminho_imagem.name}

    imagem = ler_imagem(caminho_imagem)
    mascara_raw = ler_imagem(caminho_mascara, cv2.IMREAD_GRAYSCALE)
    if imagem is None or mascara_raw is None:
        registro["classificacao"] = REVISAO_NECESSARIA
        registro["motivo"] = "falha ao ler imagem ou mascara"
        return None, registro

    if mascara_raw.shape[:2] != imagem.shape[:2]:
        mascara_raw = cv2.resize(
            mascara_raw, (imagem.shape[1], imagem.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    original = binarizar(mascara_raw)
    area_original = int(original.sum())
    registro["area_original_px"] = area_original

    if area_original == 0:
        # Sem região de partida não há bbox, nem ROI, nem referência de
        # pele. A máscara é reproduzida como está, e o caso é sinalizado.
        registro["classificacao"] = REVISAO_NECESSARIA
        registro["motivo"] = "mascara original vazia"
        registro["area_nova_px"] = 0
        registro["area_adicionada_px"] = 0
        return original, registro

    x0, y0, x1, y1 = calcular_roi(original)
    registro["roi"] = "%d,%d,%d,%d" % (x0, y0, x1, y1)

    roi_mask = np.zeros_like(original)
    roi_mask[y0:y1, x0:x1] = 1

    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Anel de pele: em volta da lesão, pulando a borda imediata, e limitado
    # à ROI para não buscar referência em outra parte do corpo.
    kernel = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dil_interno = cv2.dilate(original, kernel(ANEL_INTERNO_PX))
    dil_externo = cv2.dilate(original, kernel(ANEL_EXTERNO_PX))
    anel = ((dil_externo > 0) & (dil_interno == 0) & (roi_mask > 0)).astype(np.uint8)

    pele = estimar_pele(lab, anel)
    if pele is None:
        registro["classificacao"] = REVISAO_NECESSARIA
        registro["motivo"] = "pixels de pele insuficientes para referencia"
        registro["area_nova_px"] = area_original
        registro["area_adicionada_px"] = 0
        return original, registro
    registro["pele_lab"] = "%.0f,%.0f,%.0f" % tuple(pele)

    # A máscara original também é referência de COR, não só geométrica: o
    # interior dela é a melhor amostra disponível de como a lesão se parece
    # NESTA imagem. Sem isso, "avermelhado e diferente da pele" captura todo
    # o eritema perilesional — pele inflamada em volta da ferida, que é
    # vermelha por definição e não é lesão.
    lesao = estimar_pele(lab, cv2.erode(original, kernel(2)) if area_original > 400 else original)
    if lesao is None:
        lesao = np.median(lab[original > 0], axis=0).astype(np.float32)
    registro["lesao_lab"] = "%.0f,%.0f,%.0f" % tuple(lesao)

    # Diferenças perceptuais (CIE76) para as duas referências.
    delta_pele = np.linalg.norm(lab - pele.reshape(1, 1, 3), axis=2)
    delta_lesao = np.linalg.norm(lab - lesao.reshape(1, 1, 3), axis=2)

    # Alcance máximo da expansão, proporcional ao tamanho da própria lesão:
    # uma ferida grande pode continuar por mais pixels que uma pequena. Sem
    # este teto, bastava um istmo de poucos pixels encostando na máscara
    # para arrastar todo o fundo da foto junto com o componente.
    limite_distancia = float(np.clip(0.25 * np.sqrt(area_original), 8, 60))
    distancia = cv2.distanceTransform((original == 0).astype(np.uint8), cv2.DIST_L2, 5)
    registro["limite_distancia_px"] = round(limite_distancia, 1)

    candidatos = (
        (roi_mask > 0)
        & (original == 0)
        & (distancia <= limite_distancia)
        & (delta_pele >= DELTA_E_MINIMO)
        & (delta_lesao <= MARGEM_LESAO * delta_pele)   # mais lesão que pele
        & cores_de_interesse(hsv, lab)
    ).astype(np.uint8)

    # --- análise espacial ------------------------------------------------- #
    area_minima = max(AREA_MINIMA_COMPONENTE_PX, FRACAO_MINIMA_COMPONENTE * area_original)
    vizinhanca = cv2.dilate(original, kernel(DISTANCIA_MAXIMA_PX))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidatos, connectivity=8)
    aceitos = np.zeros_like(original)
    n_aceitos = 0
    for rotulo in range(1, n_labels):
        if stats[rotulo, cv2.CC_STAT_AREA] < area_minima:
            continue
        componente = labels == rotulo
        if not np.any(componente & (vizinhanca > 0)):
            continue  # longe da lesão: não é continuação
        aceitos[componente] = 1
        n_aceitos += 1

    nova = ((original | aceitos) > 0).astype(np.uint8)

    # Preenche buracos internos sem tocar no contorno externo, para não
    # deixar furos entre a máscara original e a região agregada.
    contornos, _ = cv2.findContours(nova, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        preenchida = np.zeros_like(nova)
        cv2.drawContours(preenchida, contornos, -1, color=1, thickness=cv2.FILLED)
        nova = preenchida

    # Garantia dura: a máscara original nunca é reduzida.
    nova = ((nova | original) > 0).astype(np.uint8)

    area_nova = int(nova.sum())
    adicionada = area_nova - area_original
    registro["area_nova_px"] = area_nova
    registro["area_adicionada_px"] = adicionada
    registro["razao_adicionada"] = round(adicionada / area_original, 4)
    registro["componentes_aceitos"] = n_aceitos

    if adicionada == 0:
        registro["classificacao"] = SEM_ALTERACAO
        registro["motivo"] = ""
    elif adicionada / area_original > RAZAO_SUSPEITA:
        registro["classificacao"] = REVISAO_NECESSARIA
        registro["motivo"] = "area adicionada desproporcional a original"
    else:
        registro["classificacao"] = MASCARA_EXPANDIDA
        registro["motivo"] = ""
    return nova, registro


def salvar_amostra(
    caminho_imagem: Path, imagem: np.ndarray, original: np.ndarray,
    nova: np.ndarray, destino: Path,
) -> None:
    """Grava um painel de conferência com 5 vistas lado a lado."""
    from utils import overlay_mask_on_image

    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    paineis = [
        ("Imagem original", rgb),
        ("Mascara original", cv2.cvtColor(original * 255, cv2.COLOR_GRAY2RGB)),
        ("Mascara nova", cv2.cvtColor(nova * 255, cv2.COLOR_GRAY2RGB)),
        ("Imagem + original", overlay_mask_on_image(rgb, original, color=(255, 0, 0))),
        ("Imagem + nova", overlay_mask_on_image(rgb, nova, color=(0, 255, 0))),
    ]
    altura = rgb.shape[0]
    faixa = np.full((26, rgb.shape[1], 3), 255, dtype=np.uint8)
    colunas = []
    for titulo, painel in paineis:
        rotulo = faixa.copy()
        cv2.putText(rotulo, titulo, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        colunas.append(np.vstack([rotulo, painel[:altura]]))
    escrever_imagem(destino, cv2.cvtColor(np.hstack(colunas), cv2.COLOR_RGB2BGR))


def recriar(
    images_dir: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    limite: Optional[int] = None,
    amostras: int = 0,
) -> Dict[str, int]:
    """Processa o dataset e grava uma nova máscara para CADA imagem.

    Args:
        images_dir: imagens; ``None`` usa ``config.IMAGES_DIR``.
        masks_dir: máscaras originais; ``None`` usa ``config.MASKS_DIR``.
        output_dir: destino; ``None`` usa ``dataset/masks_recriadas``.
        limite: processa apenas as N primeiras imagens.
        amostras: quantos painéis de conferência gravar.

    Returns:
        Contagem por classificação.
    """
    images_dir = images_dir or config.IMAGES_DIR
    masks_dir = masks_dir or config.MASKS_DIR
    output_dir = output_dir or SAIDA_PADRAO
    output_dir.mkdir(parents=True, exist_ok=True)

    imagens = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if limite:
        imagens = imagens[:limite]

    mascaras = {
        p.stem.lower(): p for p in masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    }

    linhas: List[dict] = []
    sem_mascara: List[str] = []
    contagem = {SEM_ALTERACAO: 0, MASCARA_EXPANDIDA: 0, REVISAO_NECESSARIA: 0}
    gravadas_amostra = 0

    if amostras:
        AMOSTRAS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        iterador = tqdm(imagens, desc="recriando máscaras")
    except ImportError:  # pragma: no cover
        iterador = imagens

    for caminho_imagem in iterador:
        caminho_mascara = mascaras.get(caminho_imagem.stem.lower())
        if caminho_mascara is None:
            sem_mascara.append(caminho_imagem.name)
            continue

        nova, registro = processar_par(caminho_imagem, caminho_mascara)
        classificacao = registro["classificacao"]
        contagem[classificacao] = contagem.get(classificacao, 0) + 1

        if nova is None:
            # Não foi possível analisar: a máscara original é reproduzida
            # sem alteração, para que toda imagem tenha uma nova máscara.
            shutil.copy2(caminho_mascara, output_dir / f"{caminho_imagem.stem}.png")
        else:
            escrever_imagem(output_dir / f"{caminho_imagem.stem}.png", (nova * 255).astype(np.uint8))

        linhas.append(registro)

        if amostras and gravadas_amostra < amostras and classificacao == MASCARA_EXPANDIDA:
            imagem = ler_imagem(caminho_imagem)
            original = binarizar(ler_imagem(caminho_mascara, cv2.IMREAD_GRAYSCALE))
            if imagem is not None and nova is not None:
                salvar_amostra(caminho_imagem, imagem, original, nova,
                               AMOSTRAS_DIR / f"{caminho_imagem.stem}_comparacao.png")
                gravadas_amostra += 1

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    campos = sorted({c for linha in linhas for c in linha})
    ordem = ["nome_imagem", "classificacao", "motivo"] + [c for c in campos if c not in ("nome_imagem", "classificacao", "motivo")]
    with open(RELATORIO, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=ordem)
        escritor.writeheader()
        escritor.writerows(linhas)

    _relatar(imagens, linhas, contagem, sem_mascara, output_dir, gravadas_amostra)
    return contagem


def _relatar(imagens, linhas, contagem, sem_mascara, output_dir, amostras) -> None:
    """Imprime o relatório de validação da recriação."""
    sep = "=" * 70
    expandidas = [l for l in linhas if l["classificacao"] == MASCARA_EXPANDIDA]
    razoes = [l["razao_adicionada"] for l in expandidas if "razao_adicionada" in l]

    print()
    print(sep)
    print("RECRIACAO DE MASCARAS -- expansao %.1fx da bounding box" % FATOR_EXPANSAO)
    print(sep)
    print("Imagens encontradas         : %d" % len(imagens))
    print("Imagens sem mascara original: %d" % len(sem_mascara))
    print("Novas mascaras gravadas     : %d" % len(linhas))
    print()
    print("Classificacao:")
    for chave in (SEM_ALTERACAO, MASCARA_EXPANDIDA, REVISAO_NECESSARIA):
        n = contagem.get(chave, 0)
        print("  %-20s %6d  (%.1f%%)" % (chave, n, 100 * n / max(len(linhas), 1)))
    if razoes:
        arr = np.array(razoes)
        print()
        print("Area adicionada nas expandidas (fracao da original):")
        print("  mediana=%.3f  media=%.3f  maximo=%.3f" % (np.median(arr), arr.mean(), arr.max()))
    motivos: Dict[str, int] = {}
    for linha in linhas:
        if linha["classificacao"] == REVISAO_NECESSARIA:
            motivos[linha.get("motivo", "")] = motivos.get(linha.get("motivo", ""), 0) + 1
    if motivos:
        print()
        print("Motivos de revisao:")
        for motivo, n in sorted(motivos.items(), key=lambda kv: -kv[1]):
            print("  %-45s %d" % (motivo, n))
    if sem_mascara:
        print()
        print("IMAGENS SEM MASCARA ORIGINAL (%d): %s" % (len(sem_mascara), ", ".join(sem_mascara[:8])))
    print()
    print("Novas mascaras : %s" % output_dir)
    print("Relatorio      : %s" % RELATORIO)
    if amostras:
        print("Amostras (%d)  : %s" % (amostras, AMOSTRAS_DIR))
    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recria as máscaras do dataset expandindo a original com análise da imagem."
    )
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--limite", type=int, default=None,
                        help="Processa apenas as N primeiras imagens.")
    parser.add_argument("--amostras", type=int, default=0,
                        help="Quantos paineis de conferencia gravar.")
    args = parser.parse_args()

    recriar(
        images_dir=Path(args.images_dir) if args.images_dir else None,
        masks_dir=Path(args.masks_dir) if args.masks_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limite=args.limite,
        amostras=args.amostras,
    )
