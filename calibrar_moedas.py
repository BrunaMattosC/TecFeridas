"""
calibrar_moedas.py
===================
Calibração pixel -> centímetro usando as moedas de ``Moedas Parâmetro``.

Para cada fotografia de moeda o script localiza a moeda, gera a máscara da
sua região, mede o diâmetro em pixels e o confronta com o diâmetro físico
oficial, produzindo os fatores ``pixels/cm`` e ``cm/pixel``.

Fonte das dimensões físicas
---------------------------
``Moedas Parâmetro/Dimensões Oficiais/Dimensões Oficiais.png`` — tabela do
projeto, transcrita em :data:`DIMENSOES_OFICIAIS`. Nenhum diâmetro é
estimado a partir da imagem.

**A tabela do projeto NÃO inclui a moeda de R$ 1,00.** O valor usado para
ela (27,00 mm) é externo — diâmetro da 2ª família do Real — e está marcado
como tal em :data:`DIMENSOES_OFICIAIS` e no relatório. A tabela também não
traz tolerâncias; a classificação usa consistência entre moedas, conforme
:data:`LIMITE_DENTRO` e :data:`LIMITE_REVISAO`.

Medição
-------
O diâmetro vem da transformada de Hough, que ajusta um círculo direto na
borda da moeda e ignora a sombra projetada — contaminação que derrubava a
circularidade da máscara para ~0,80 quando a segmentação era feita só por
diferença de cor. A máscara por cor é mantida para área, perímetro e
controle de qualidade, e sua divergência em relação a Hough é um dos
critérios de revisão.

Uso:
    python calibrar_moedas.py              # processa tudo
    python calibrar_moedas.py --limite 40  # amostra rápida
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from utils import escrever_imagem, get_logger, ler_imagem

logger = get_logger(__name__, config.LOGS_DIR / "calibrar_moedas.log")

MOEDAS_DIR = config.BASE_DIR / "Moedas Parâmetro"
TABELA_OFICIAL = MOEDAS_DIR / "Dimensões Oficiais" / "Dimensões Oficiais.png"

RESULTADOS_DIR = config.BASE_DIR / "Resultados"
MOEDAS_SAIDA = RESULTADOS_DIR / "Moedas"
MASCARAS_DIR = MOEDAS_SAIDA / "Mascaras"
MEDICOES_CSV = MOEDAS_SAIDA / "Medicoes" / "medicoes_moedas.csv"
CALIBRACAO_CSV = MOEDAS_SAIDA / "Calibracao" / "calibracao_por_moeda.csv"
COMPARACOES_DIR = MOEDAS_SAIDA / "Comparacoes"


@dataclass(frozen=True)
class MoedaOficial:
    """Dimensões físicas oficiais de uma moeda."""

    centavos: int
    rotulo: str
    diametro_mm: float
    espessura_mm: float
    peso_g: float
    fonte: str

    @property
    def diametro_cm(self) -> float:
        return self.diametro_mm / 10.0


_FONTE_PROJETO = "Moedas Parâmetro/Dimensões Oficiais/Dimensões Oficiais.png"
_FONTE_EXTERNA = "EXTERNA - 2a familia do Real (ausente na tabela do projeto)"

# Transcrição literal da tabela do projeto. A linha de R$ 0,01 existe na
# tabela mas não tem imagens correspondentes, e por isso não é usada.
DIMENSOES_OFICIAIS: Dict[int, MoedaOficial] = {
    5: MoedaOficial(5, "R$ 0,05", 21.00, 1.20, 3.27, _FONTE_PROJETO),
    10: MoedaOficial(10, "R$ 0,10", 22.00, 1.20, 3.59, _FONTE_PROJETO),
    25: MoedaOficial(25, "R$ 0,25", 23.50, 1.40, 4.78, _FONTE_PROJETO),
    50: MoedaOficial(50, "R$ 0,50", 23.00, 1.20, 3.92, _FONTE_PROJETO),
    100: MoedaOficial(100, "R$ 1,00", 27.00, 1.95, 7.00, _FONTE_EXTERNA),
}

# Classificação por consistência: desvio da escala de cada imagem em
# relação à escala mediana do conjunto. Não são tolerâncias oficiais de
# cunhagem — a tabela do projeto não as fornece.
LIMITE_DENTRO = 5.0    # % de desvio
LIMITE_REVISAO = 10.0  # % de desvio

DENTRO = "DENTRO_DA_TOLERANCIA"
FORA = "FORA_DA_TOLERANCIA"
REVISAO = "REVISAO_NECESSARIA"

# Hough roda em imagem reduzida por este fator; o raio é reescalado depois.
ESCALA_HOUGH = 4
# Razão mínima entre eixos da elipse ajustada. Abaixo disso a moeda está
# inclinada o bastante para o diâmetro projetado deixar de ser confiável.
RAZAO_EIXOS_MINIMA = 0.85
# Divergência máxima tolerada entre o diâmetro de Hough e o da máscara.
DIVERGENCIA_MAXIMA_PCT = 12.0


def detectar_circulo(imagem: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Localiza a moeda por transformada de Hough.

    Returns:
        ``(cx, cy, raio)`` em pixels da imagem original, ou ``None``.
    """
    altura, largura = imagem.shape[:2]
    reduzida = cv2.resize(
        imagem, (largura // ESCALA_HOUGH, altura // ESCALA_HOUGH), interpolation=cv2.INTER_AREA
    )
    cinza = cv2.medianBlur(cv2.cvtColor(reduzida, cv2.COLOR_BGR2GRAY), 5)
    h_red = cinza.shape[0]

    circulos = cv2.HoughCircles(
        cinza, cv2.HOUGH_GRADIENT, dp=1, minDist=h_red // 4,
        param1=100, param2=30,
        minRadius=int(0.03 * h_red), maxRadius=int(0.22 * h_red),
    )
    if circulos is None:
        return None
    cx, cy, raio = circulos[0][0]
    return float(cx) * ESCALA_HOUGH, float(cy) * ESCALA_HOUGH, float(raio) * ESCALA_HOUGH


def segmentar_moeda(
    imagem: np.ndarray, centro: Tuple[float, float], raio: float
) -> Optional[np.ndarray]:
    """Gera a máscara da moeda, restrita à vizinhança do círculo detectado.

    Restringir a busca a um disco um pouco maior que a moeda mantém a
    sombra projetada fora do alcance da limiarização, que de outro modo a
    incorporaria e deformaria a máscara.
    """
    altura, largura = imagem.shape[:2]
    cx, cy = centro
    janela = np.zeros((altura, largura), dtype=np.uint8)
    cv2.circle(janela, (int(round(cx)), int(round(cy))), int(round(raio * 1.25)), 1, -1)

    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB).astype(np.float32)
    anel = (janela > 0) & (
        np.linalg.norm(
            np.stack(np.meshgrid(np.arange(largura), np.arange(altura)), axis=-1)
            - np.array([cx, cy]), axis=-1
        ) > raio * 1.10
    )
    if anel.sum() < 100:
        return None
    fundo = np.median(lab[anel], axis=0)

    delta = np.linalg.norm(lab - fundo.reshape(1, 1, 3), axis=2)
    delta8 = np.clip(delta * 2, 0, 255).astype(np.uint8)
    delta8[janela == 0] = 0
    # Otsu calculado SÓ com os pixels da janela: incluir a imagem inteira
    # deslocaria o limiar, porque o fundo domina o histograma.
    limiar, _ = cv2.threshold(delta8[janela > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = ((delta8 >= limiar) & (janela > 0)).astype(np.uint8)

    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)

    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n_labels <= 1:
        return None
    rotulo_centro = labels[int(round(cy)), int(round(cx))]
    if rotulo_centro == 0:  # centro caiu no fundo: usa o maior componente
        areas = [(labels == i).sum() for i in range(1, n_labels)]
        rotulo_centro = 1 + int(np.argmax(areas))
    moeda = (labels == rotulo_centro).astype(np.uint8)

    contornos, _ = cv2.findContours(moeda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    preenchida = np.zeros_like(moeda)
    cv2.drawContours(preenchida, contornos, -1, color=1, thickness=cv2.FILLED)
    return preenchida


def medir_moeda(caminho: Path, centavos: int, lado: str) -> Optional[dict]:
    """Processa uma fotografia de moeda e devolve o registro de medição."""
    imagem = ler_imagem(caminho)
    if imagem is None or imagem.ndim != 3:
        return {"arquivo": caminho.name, "centavos": centavos, "lado": lado,
                "situacao": REVISAO, "observacao": "imagem ilegivel"}

    altura, largura = imagem.shape[:2]
    oficial = DIMENSOES_OFICIAIS[centavos]

    circulo = detectar_circulo(imagem)
    if circulo is None:
        return {"arquivo": caminho.name, "centavos": centavos, "lado": lado,
                "situacao": REVISAO, "observacao": "moeda nao localizada"}
    cx, cy, raio = circulo
    diametro_px = 2.0 * raio

    problemas: List[str] = []
    margem = raio * 1.05
    if cx - margem < 0 or cy - margem < 0 or cx + margem > largura or cy + margem > altura:
        problemas.append("moeda cortada pela borda")

    mascara = segmentar_moeda(imagem, (cx, cy), raio)
    area_px = perimetro_px = circularidade = razao_eixos = float("nan")
    d_mascara = float("nan")
    bbox = ""
    if mascara is not None and mascara.sum() > 0:
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno = max(contornos, key=cv2.contourArea)
        area_px = float(mascara.sum())
        perimetro_px = float(cv2.arcLength(contorno, True))
        circularidade = (4 * np.pi * area_px / (perimetro_px ** 2)) if perimetro_px else 0.0
        d_mascara = 2.0 * float(np.sqrt(area_px / np.pi))
        x, y, w, h = cv2.boundingRect(contorno)
        bbox = "%d,%d,%d,%d" % (x, y, w, h)
        if len(contorno) >= 5:
            _, (menor, maior), _ = cv2.fitEllipse(contorno)
            razao_eixos = float(min(menor, maior) / max(menor, maior))
    else:
        problemas.append("falha ao segmentar a mascara")

    if not np.isnan(razao_eixos) and razao_eixos < RAZAO_EIXOS_MINIMA:
        problemas.append("moeda inclinada (razao de eixos %.2f)" % razao_eixos)
    divergencia = float("nan")
    if not np.isnan(d_mascara):
        divergencia = 100.0 * abs(d_mascara - diametro_px) / diametro_px
        if divergencia > DIVERGENCIA_MAXIMA_PCT:
            problemas.append("Hough e mascara divergem %.0f%%" % divergencia)

    ppc = diametro_px / oficial.diametro_cm
    cpp = oficial.diametro_cm / diametro_px

    return {
        "arquivo": caminho.name,
        "centavos": centavos,
        "rotulo": oficial.rotulo,
        "lado": lado,
        "fonte_dimensao": oficial.fonte,
        "diametro_oficial_mm": oficial.diametro_mm,
        "largura_img_px": largura,
        "altura_img_px": altura,
        "centro_x": round(cx, 1),
        "centro_y": round(cy, 1),
        "diametro_px": round(diametro_px, 2),
        "diametro_mascara_px": round(d_mascara, 2) if not np.isnan(d_mascara) else "",
        "divergencia_pct": round(divergencia, 2) if not np.isnan(divergencia) else "",
        "area_px2": round(area_px, 1) if not np.isnan(area_px) else "",
        "perimetro_px": round(perimetro_px, 1) if not np.isnan(perimetro_px) else "",
        "bbox": bbox,
        "circularidade": round(circularidade, 4) if not np.isnan(circularidade) else "",
        "razao_eixos": round(razao_eixos, 4) if not np.isnan(razao_eixos) else "",
        "pixels_por_cm": round(ppc, 3),
        "cm_por_pixel": round(cpp, 8),
        "observacao": "; ".join(problemas),
        "_mascara": mascara,
        "_problemas": problemas,
    }


def classificar(registros: List[dict]) -> float:
    """Classifica cada medição pelo desvio da escala mediana do conjunto.

    Returns:
        A escala mediana (pixels/cm) usada como referência.
    """
    escalas = [r["pixels_por_cm"] for r in registros if "pixels_por_cm" in r]
    mediana = float(np.median(escalas)) if escalas else float("nan")
    for r in registros:
        if "pixels_por_cm" not in r:
            continue
        desvio = 100.0 * abs(r["pixels_por_cm"] - mediana) / mediana
        r["desvio_da_mediana_pct"] = round(desvio, 2)
        if r["_problemas"]:
            r["situacao"] = REVISAO
        elif desvio <= LIMITE_DENTRO:
            r["situacao"] = DENTRO
        elif desvio <= LIMITE_REVISAO:
            r["situacao"] = REVISAO
        else:
            r["situacao"] = FORA
    return mediana


def calibrar(limite: Optional[int] = None, salvar_mascaras: int = 10) -> dict:
    """Processa todas as fotografias de moedas e produz a calibração.

    Args:
        limite: processa apenas as N primeiras imagens de cada moeda.
        salvar_mascaras: quantas máscaras gravar por moeda (amostra visual).

    Returns:
        Dicionário com a escala mediana global e as escalas por moeda.
    """
    if not MOEDAS_DIR.exists():
        raise FileNotFoundError(f"Pasta de moedas não encontrada: {MOEDAS_DIR}")
    if not TABELA_OFICIAL.exists():
        raise FileNotFoundError(f"Tabela oficial não encontrada: {TABELA_OFICIAL}")

    for pasta in (MASCARAS_DIR, MEDICOES_CSV.parent, CALIBRACAO_CSV.parent, COMPARACOES_DIR):
        pasta.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    registros: List[dict] = []
    for centavos in sorted(DIMENSOES_OFICIAIS):
        pasta_moeda = MOEDAS_DIR / str(centavos)
        if not pasta_moeda.exists():
            logger.warning("Pasta ausente para %d centavos: %s", centavos, pasta_moeda)
            continue
        for lado in ("front", "back"):
            arquivos = sorted(p for p in (pasta_moeda / lado).glob("*") if p.is_file())
            if limite:
                arquivos = arquivos[:limite]
            salvos = 0
            for caminho in tqdm(arquivos, desc="%3d %s" % (centavos, lado)):
                registro = medir_moeda(caminho, centavos, lado)
                if registro is None:
                    continue
                mascara = registro.pop("_mascara", None)
                if mascara is not None and salvos < salvar_mascaras:
                    escrever_imagem(
                        MASCARAS_DIR / ("%s.png" % caminho.stem), (mascara * 255).astype(np.uint8)
                    )
                    salvos += 1
                registros.append(registro)

    validos = [r for r in registros if "pixels_por_cm" in r]
    mediana_global = classificar(validos)

    campos = [c for c in validos[0] if not c.startswith("_")] if validos else []
    for extra in ("desvio_da_mediana_pct", "situacao"):
        if extra not in campos:
            campos.append(extra)
    with open(MEDICOES_CSV, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        for r in registros:
            escritor.writerow({k: v for k, v in r.items() if not k.startswith("_")})

    por_moeda = _resumo_por_moeda(validos)
    with open(CALIBRACAO_CSV, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(por_moeda[0]))
        escritor.writeheader()
        escritor.writerows(por_moeda)

    _relatar(registros, validos, por_moeda, mediana_global)
    return {"mediana_global_ppc": mediana_global, "por_moeda": por_moeda}


def _resumo_por_moeda(validos: List[dict]) -> List[dict]:
    """Agrega as medições por denominação."""
    linhas = []
    for centavos in sorted(DIMENSOES_OFICIAIS):
        grupo = [r for r in validos if r["centavos"] == centavos]
        if not grupo:
            continue
        oficial = DIMENSOES_OFICIAIS[centavos]
        diams = np.array([r["diametro_px"] for r in grupo])
        ppcs = np.array([r["pixels_por_cm"] for r in grupo])
        linhas.append({
            "centavos": centavos,
            "rotulo": oficial.rotulo,
            "fonte_dimensao": oficial.fonte,
            "diametro_oficial_mm": oficial.diametro_mm,
            "n_imagens": len(grupo),
            "diametro_px_mediano": round(float(np.median(diams)), 2),
            "diametro_px_desvio": round(float(diams.std()), 2),
            "pixels_por_cm_mediano": round(float(np.median(ppcs)), 3),
            "pixels_por_cm_p25": round(float(np.percentile(ppcs, 25)), 3),
            "pixels_por_cm_p75": round(float(np.percentile(ppcs, 75)), 3),
            "cm_por_pixel_mediano": round(1.0 / float(np.median(ppcs)), 8),
            "dispersao_relativa_pct": round(100.0 * float(ppcs.std()) / float(np.median(ppcs)), 2),
            "dentro": sum(1 for r in grupo if r.get("situacao") == DENTRO),
            "revisao": sum(1 for r in grupo if r.get("situacao") == REVISAO),
            "fora": sum(1 for r in grupo if r.get("situacao") == FORA),
        })
    return linhas


def _relatar(registros, validos, por_moeda, mediana_global) -> None:
    """Imprime o resumo da calibração."""
    sep = "=" * 78
    print()
    print(sep)
    print("CALIBRACAO POR MOEDAS")
    print(sep)
    print("Imagens processadas : %d" % len(registros))
    print("Medicoes validas    : %d" % len(validos))
    print("Falhas de deteccao  : %d" % (len(registros) - len(validos)))
    print("Escala mediana      : %.2f px/cm  (%.6f cm/px)" % (mediana_global, 1 / mediana_global))
    print()
    print("%-9s %-22s %5s %11s %12s %10s %7s %7s %6s" % (
        "MOEDA", "FONTE DA DIMENSAO", "N", "D_OFIC(mm)", "D_MED(px)", "PX/CM", "DENTRO", "REVISAO", "FORA"))
    for linha in por_moeda:
        fonte = "projeto" if linha["fonte_dimensao"].startswith("Moedas") else "EXTERNA"
        print("%-9s %-22s %5d %11.2f %12.1f %10.2f %7d %7d %6d" % (
            linha["rotulo"], fonte, linha["n_imagens"], linha["diametro_oficial_mm"],
            linha["diametro_px_mediano"], linha["pixels_por_cm_mediano"],
            linha["dentro"], linha["revisao"], linha["fora"]))
    print()
    print("Mascaras  : %s" % MASCARAS_DIR)
    print("Medicoes  : %s" % MEDICOES_CSV)
    print("Calibracao: %s" % CALIBRACAO_CSV)
    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibração pixel->cm pelas moedas de referência.")
    parser.add_argument("--limite", type=int, default=None,
                        help="Processa apenas as N primeiras imagens de cada moeda/lado.")
    parser.add_argument("--mascaras-por-moeda", type=int, default=10,
                        help="Quantas mascaras gravar por moeda/lado.")
    args = parser.parse_args()
    calibrar(limite=args.limite, salvar_mascaras=args.mascaras_por_moeda)
