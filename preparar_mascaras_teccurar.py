"""
preparar_mascaras_teccurar.py
==============================
Deriva máscaras binárias a partir das fotos anotadas em ``TecCurar_NewMasks``.

O que são os arquivos de entrada
--------------------------------
``TecCurar_NewMasks`` NÃO contém máscaras: contém as fotos clínicas de
``Fotos TecFeridas`` com a região da ferida pintada por cima em verde
sólido. Medido nos 32 arquivos, o verde é uma cor chapada e estável —
HSV H=65+-1, S=157+-5, V=215+-3 — degradada apenas pela compressão JPEG.

Este script extrai essa região e grava uma máscara binária de verdade
(PNG, 1 canal, valores 0/255), no formato que o restante do pipeline já
consome. Os arquivos de ``TecCurar_NewMasks`` e de ``Fotos TecFeridas``
são tratados como somente leitura e nunca são sobrescritos.

O par gerado é (foto ORIGINAL sem anotação, máscara derivada). Usar a
foto anotada como entrada de treino vazaria o rótulo para dentro da
própria imagem: o modelo aprenderia a achar tinta verde, não ferida.

Saída
-----
    dataset_teccurar/
        images/        cópia byte a byte da foto original (.jpeg)
        masks/         máscara binária derivada (.png)
        manifesto.csv  rastreabilidade e métricas de cada par

Uso:
    python preparar_mascaras_teccurar.py
    python preparar_mascaras_teccurar.py --verificar   # só confere, não grava
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from measure import clean_mask
from utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR / "preparar_mascaras_teccurar.log")

ANOTADAS_DIR = config.BASE_DIR / "TecCurar_NewMasks"
FOTOS_DIR = config.BASE_DIR / "Fotos TecFeridas"
SAIDA_DIR = config.BASE_DIR / "dataset_teccurar"
IMAGENS_SAIDA = SAIDA_DIR / "images"
MASCARAS_SAIDA = SAIDA_DIR / "masks"
MANIFESTO = SAIDA_DIR / "manifesto.csv"

# Faixa HSV da tinta de anotação. Larga o suficiente para absorver o
# serrilhado que o JPEG cria na borda do traço, e ainda assim distante de
# qualquer tom de pele ou tecido: aplicada às 508 fotos NÃO anotadas,
# captura 0,02% dos pixels em média.
VERDE_HSV_MIN = np.array([55, 80, 120], dtype=np.uint8)
VERDE_HSV_MAX = np.array([75, 255, 255], dtype=np.uint8)

# Área mínima da anotação, em fração da imagem. Abaixo disso o resultado é
# quase certamente respingo de compressão, não marcação do avaliador.
AREA_MINIMA_RELATIVA = 0.0005


def ler_imagem(caminho: Path) -> Optional[np.ndarray]:
    """Lê uma imagem tolerando acentos no caminho.

    ``cv2.imread`` falha silenciosamente com caracteres não-ASCII no
    Windows (devolve ``None``) — e todos os nomes aqui têm acento.
    ``fromfile`` + ``imdecode`` contorna, lendo os bytes pelo Python.
    """
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_UNCHANGED)


def extrair_mascara_verde(anotada: np.ndarray) -> np.ndarray:
    """Extrai a região pintada de verde e devolve máscara binária 0/1.

    A limpeza reaproveita :func:`measure.clean_mask`, a mesma rotina que o
    projeto já aplica antes de medir: preenche buracos internos (falhas de
    compressão dentro do traço) e descarta componentes irrelevantes.
    """
    hsv = cv2.cvtColor(anotada, cv2.COLOR_BGR2HSV)
    bruta = cv2.inRange(hsv, VERDE_HSV_MIN, VERDE_HSV_MAX)
    return clean_mask(bruta)


def _assinatura(img: np.ndarray, altura: int = 96, largura: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    """Reduz a imagem e devolve (miniatura, máscara do que não é anotação)."""
    pequena = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA).astype(np.int16)
    hsv = cv2.cvtColor(pequena.astype(np.uint8), cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, VERDE_HSV_MIN, VERDE_HSV_MAX) > 0
    return pequena, ~verde


def localizar_original(
    anotada_path: Path, anotada_img: np.ndarray, catalogo: Dict[str, List[Path]],
    miniaturas: Dict[Path, np.ndarray],
) -> Tuple[Optional[Path], str, float]:
    """Encontra a foto original correspondente a uma foto anotada.

    Duas estratégias, nesta ordem — nunca por posição na listagem:

    1. **Nome idêntico** em ``Fotos TecFeridas``. Vale para as categorias
       cujos arquivos mantêm o nome (Deiscência, Queimaduras, Supra, Úlcera).
    2. **Conteúdo**: compara a imagem fora da área pintada contra todas as
       fotos originais. Necessário para "Pé diabético - N", cujos originais
       têm nomes sem relação (``IMG_*.jpeg`` e UUIDs).

    Returns:
        Tupla ``(caminho_original, metodo, distancia)``. ``caminho_original``
        é ``None`` quando nenhuma correspondência confiável é encontrada.
    """
    candidatos = catalogo.get(anotada_path.name)
    if candidatos:
        return candidatos[0], "nome", 0.0

    referencia, valido = _assinatura(anotada_img)
    melhor, menor = None, float("inf")
    for caminho, mini in miniaturas.items():
        distancia = float(np.abs(referencia - mini).sum(axis=2)[valido].mean())
        if distancia < menor:
            menor, melhor = distancia, caminho
    return melhor, "conteudo", menor


def gerar_nome(anotada_path: Path) -> str:
    """Converte o nome da anotação em um identificador ASCII estável.

    Acentos e espaços são removidos de propósito: o ``cv2.imread`` usado
    pelo pipeline não abre caminhos não-ASCII no Windows, então um par
    chamado "Pé diabético - 1" seria ilegível para o próprio projeto. O
    nome original fica registrado no manifesto.
    """
    texto = unicodedata.normalize("NFKD", anotada_path.stem)
    texto = texto.encode("ascii", "ignore").decode("ascii").lower()
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    # "pe_diabetico_1" -> "pe_diabetico_001", para ordenar corretamente
    return re.sub(r"_(\d+)$", lambda m: "_%03d" % int(m.group(1)), texto)


def preparar(gravar: bool = True) -> int:
    """Executa a derivação completa e imprime o relatório de verificação.

    Args:
        gravar: quando ``False``, apenas analisa e relata, sem escrever nada.

    Returns:
        0 se todos os pares foram resolvidos; 1 se houve alguma pendência.
    """
    if not ANOTADAS_DIR.exists():
        raise FileNotFoundError(f"Pasta de anotações não encontrada: {ANOTADAS_DIR}")
    if not FOTOS_DIR.exists():
        raise FileNotFoundError(f"Pasta de fotos originais não encontrada: {FOTOS_DIR}")

    anotadas = sorted(ANOTADAS_DIR.glob("*.jpeg"))
    if not anotadas:
        raise FileNotFoundError(f"Nenhuma anotação .jpeg em {ANOTADAS_DIR}")

    # Catálogo das fotos originais: por nome e por miniatura (para o casamento
    # por conteúdo). As fotos são lidas uma única vez.
    catalogo: Dict[str, List[Path]] = {}
    miniaturas: Dict[Path, np.ndarray] = {}
    for caminho in FOTOS_DIR.rglob("*.jpeg"):
        if not caminho.is_file():
            continue
        catalogo.setdefault(caminho.name, []).append(caminho)
        img = ler_imagem(caminho)
        if img is not None and img.ndim == 3:
            miniaturas[caminho] = _assinatura(img)[0]

    logger.info("Anotações: %d | fotos originais indexadas: %d", len(anotadas), len(miniaturas))

    if gravar:
        IMAGENS_SAIDA.mkdir(parents=True, exist_ok=True)
        MASCARAS_SAIDA.mkdir(parents=True, exist_ok=True)

    linhas: List[dict] = []
    sem_original: List[str] = []
    sem_anotacao: List[str] = []
    usados: Dict[Path, str] = {}

    for anotada_path in anotadas:
        anotada = ler_imagem(anotada_path)
        if anotada is None or anotada.ndim != 3:
            sem_original.append("%s (ilegível)" % anotada_path.name)
            continue

        original_path, metodo, distancia = localizar_original(
            anotada_path, anotada, catalogo, miniaturas
        )
        if original_path is None:
            sem_original.append(anotada_path.name)
            continue

        original = ler_imagem(original_path)
        if original is None:
            sem_original.append("%s (original ilegível)" % anotada_path.name)
            continue

        mascara = extrair_mascara_verde(anotada)

        # Algumas anotações foram salvas em escala reduzida. Enquanto a
        # proporção for a mesma, a máscara pode ser levada de volta à
        # resolução da foto original sem distorcer a geometria da ferida;
        # INTER_NEAREST preserva o caráter binário, sem criar meio-tom.
        if original.shape[:2] != anotada.shape[:2]:
            escala_y = original.shape[0] / anotada.shape[0]
            escala_x = original.shape[1] / anotada.shape[1]
            if abs(escala_y - escala_x) > 0.01:
                sem_original.append(
                    "%s (proporção incompatível: %s vs %s)"
                    % (anotada_path.name, anotada.shape[:2], original.shape[:2])
                )
                continue
            mascara = cv2.resize(
                mascara, (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            anotada = cv2.resize(
                anotada, (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            logger.info(
                "%s estava em escala %.4f; máscara reamostrada para %dx%d.",
                anotada_path.name, escala_y, original.shape[1], original.shape[0],
            )
        area = float(mascara.mean())
        if area < AREA_MINIMA_RELATIVA:
            sem_anotacao.append("%s (área verde %.4f%%)" % (anotada_path.name, 100 * area))
            continue

        # Validação independente: a região pintada é, por definição, onde a
        # foto anotada difere da original. Se a máscara extraída não bate com
        # essa diferença, a extração por cor errou.
        #
        # O limiar é 100 (soma dos 3 canais, escala 0-765) para ignorar a
        # recompressão JPEG: arquivos salvos novamente com qualidade menor
        # diferem da original em ~50 por toda a imagem, ruído difuso que a
        # 40 derrubava o IoU de máscaras comprovadamente corretas. Um traço
        # de tinta de verdade produz diferença de centenas.
        diferenca = np.abs(anotada.astype(np.int16) - original.astype(np.int16)).sum(axis=2) > 100
        intersecao = float(np.logical_and(mascara > 0, diferenca).sum())
        uniao = float(np.logical_or(mascara > 0, diferenca).sum())
        iou = intersecao / uniao if uniao else 0.0

        nome = gerar_nome(anotada_path)
        if original_path in usados:
            logger.warning(
                "Foto original %s já usada por %s; %s aponta para a mesma.",
                original_path.name, usados[original_path], anotada_path.name,
            )
        usados[original_path] = nome

        if gravar:
            # A imagem é copiada byte a byte: nenhuma recompressão, nenhuma
            # alteração de pixel no material original.
            shutil.copy2(original_path, IMAGENS_SAIDA / ("%s%s" % (nome, original_path.suffix.lower())))
            destino_mascara = MASCARAS_SAIDA / ("%s.png" % nome)
            ok, buffer = cv2.imencode(".png", (mascara * 255).astype(np.uint8))
            if not ok:
                raise RuntimeError("Falha ao codificar a máscara de %s" % anotada_path.name)
            buffer.tofile(str(destino_mascara))

        linhas.append({
            "nome": nome,
            "anotacao_origem": anotada_path.name,
            "foto_original": str(original_path.relative_to(config.BASE_DIR)),
            "metodo_correspondencia": metodo,
            "distancia_conteudo": round(distancia, 2),
            "altura": anotada.shape[0],
            "largura": anotada.shape[1],
            "area_mascara_px": int((mascara > 0).sum()),
            "area_mascara_percent": round(100 * area, 3),
            "iou_contra_diferenca": round(iou, 4),
        })

    if gravar and linhas:
        with open(MANIFESTO, "w", newline="", encoding="utf-8") as fh:
            escritor = csv.DictWriter(fh, fieldnames=list(linhas[0]))
            escritor.writeheader()
            escritor.writerows(linhas)
        logger.info("Manifesto salvo em %s", MANIFESTO)

    _relatar(anotadas, linhas, sem_original, sem_anotacao, usados, gravar)
    return 0 if not (sem_original or sem_anotacao) else 1


def _relatar(anotadas, linhas, sem_original, sem_anotacao, usados, gravou) -> None:
    """Imprime o relatório de verificação da migração."""
    sep = "=" * 66
    ious = [linha["iou_contra_diferenca"] for linha in linhas]
    por_metodo = {}
    for linha in linhas:
        por_metodo[linha["metodo_correspondencia"]] = por_metodo.get(linha["metodo_correspondencia"], 0) + 1

    print()
    print(sep)
    print("DERIVACAO DE MASCARAS -- TecCurar_NewMasks")
    print(sep)
    print("Anotacoes encontradas            : %d" % len(anotadas))
    print("Pares gerados                    : %d" % len(linhas))
    print("  correspondencia por nome       : %d" % por_metodo.get("nome", 0))
    print("  correspondencia por conteudo   : %d" % por_metodo.get("conteudo", 0))
    print("Fotos originais distintas usadas : %d" % len(usados))
    print("Anotacoes sem foto original      : %d" % len(sem_original))
    print("Anotacoes sem area verde valida  : %d" % len(sem_anotacao))
    if ious:
        print()
        print("Concordancia mascara x regiao alterada (IoU):")
        print("  minimo  : %.4f" % min(ious))
        print("  mediana : %.4f" % float(np.median(ious)))
        print("  maximo  : %.4f" % max(ious))
        baixos = [linha for linha in linhas if linha["iou_contra_diferenca"] < 0.80]
        print("  abaixo de 0,80 : %d" % len(baixos))
        for linha in baixos:
            print("     %-28s IoU=%.3f" % (linha["nome"], linha["iou_contra_diferenca"]))
    for titulo, itens in (("ANOTACOES SEM FOTO ORIGINAL", sem_original),
                          ("ANOTACOES SEM AREA VERDE VALIDA", sem_anotacao)):
        if itens:
            print()
            print("%s (%d):" % (titulo, len(itens)))
            for item in itens:
                print("   - %s" % item)
    print()
    if gravou:
        print("Imagens : %s" % IMAGENS_SAIDA)
        print("Mascaras: %s" % MASCARAS_SAIDA)
        print("Manifesto: %s" % MANIFESTO)
    else:
        print("MODO VERIFICACAO -- nenhum arquivo foi gravado.")
    print(sep)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deriva máscaras binárias das fotos anotadas em TecCurar_NewMasks."
    )
    parser.add_argument("--verificar", action="store_true",
                        help="Apenas analisa e relata, sem gravar nenhum arquivo.")
    args = parser.parse_args()
    raise SystemExit(preparar(gravar=not args.verificar))
