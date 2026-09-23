"""
conftest.py
===========
Configuração comum dos testes: garante que a raiz do projeto esteja no
``sys.path`` (os módulos vivem na raiz, sem pacote instalável) e expõe
fábricas de máscaras sintéticas usadas pelos testes de medição.

As máscaras são geradas analiticamente para que os valores esperados de
área, comprimento, largura e perímetro sejam conhecidos, permitindo
validar a medição sem depender de um modelo treinado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402  (import após ajuste do sys.path)

IMAGE_SIZE = 400


@pytest.fixture
def empty_mask() -> np.ndarray:
    """Máscara totalmente vazia (nenhuma ferida)."""
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


@pytest.fixture
def single_rect_mask() -> np.ndarray:
    """Uma única ferida retangular de 120x60 px em (100, 80).

    Área esperada: 120 * 60 = 7200 px².
    Comprimento esperado: 120 px; largura esperada: 60 px.
    """
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[80:140, 100:220] = 1  # linhas 80..139 (60), colunas 100..219 (120)
    return mask


@pytest.fixture
def single_circle_mask() -> np.ndarray:
    """Uma única ferida circular de raio 70 px centrada em (200, 200)."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cv2.circle(mask, (200, 200), 70, 1, thickness=-1)
    return mask


@pytest.fixture
def two_wounds_mask() -> np.ndarray:
    """Duas feridas bem separadas, de áreas diferentes.

    Ferida maior: retângulo 100x80 (8000 px²).
    Ferida menor: retângulo 60x40 (2400 px²).
    """
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[30:110, 30:130] = 1   # 80 linhas x 100 colunas = 8000
    mask[250:290, 250:310] = 1  # 40 linhas x 60 colunas = 2400
    return mask


@pytest.fixture
def irregular_mask() -> np.ndarray:
    """Ferida em formato de "L" — côncava, com solidez claramente < 1."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[100:260, 100:160] = 1  # braço vertical
    mask[200:260, 100:280] = 1  # braço horizontal
    return mask


@pytest.fixture
def noisy_mask(single_rect_mask: np.ndarray) -> np.ndarray:
    """Ferida principal acompanhada de ruído de segmentação.

    Acrescenta pixels isolados e blocos minúsculos (1-3 px) longe da
    ferida, além de um buraco interno de 6x6 px.
    """
    mask = single_rect_mask.copy()
    # ruído "sal": pixels isolados
    for y, x in [(10, 10), (12, 300), (350, 40), (370, 380), (20, 200)]:
        mask[y, x] = 1
    # blocos minúsculos de 2x2 e 3x3
    mask[300:302, 100:102] = 1
    mask[320:323, 340:343] = 1
    # buraco interno na ferida
    mask[100:106, 150:156] = 0
    return mask


@pytest.fixture
def rotated_rect_mask() -> np.ndarray:
    """Retângulo 140x50 rotacionado 30° — testa invariância à rotação."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    rect = ((200.0, 200.0), (140.0, 50.0), 30.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], 1)
    return mask
