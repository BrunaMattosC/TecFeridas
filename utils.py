"""
utils.py
========
Funções utilitárias compartilhadas por todos os scripts do projeto:

- fixação de seeds para reprodutibilidade;
- configuração de logging;
- funções de visualização (overlay de máscara, grids de exemplos);
- helpers genéricos de I/O.

``torch`` é importado sob demanda (dentro de :func:`seed_everything`)
para que as rotinas puramente de imagem — usadas pela medição em
:mod:`measure` — possam ser importadas e testadas sem carregar toda a
pilha de deep learning.

A escolha de dispositivo (GPU/CPU) não vive aqui: ela é resolvida por
:func:`hardware.resolver_dispositivo`, o único ponto de decisão do
projeto.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #

def seed_everything(seed: int = 42) -> None:
    """Fixa as seeds de Python, NumPy e PyTorch (CPU e GPU).

    Args:
        seed: valor inteiro usado para inicializar todos os geradores
            de números aleatórios.
    """
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinismo (pode reduzir performance, mas garante reprodutibilidade).
    # O comportamento padrão do projeto é inalterado: determinístico. Só
    # quem ligar CUDNN_BENCHMARK=1 explicitamente troca reprodutibilidade
    # por velocidade — ver config.DeviceConfig.cudnn_benchmark.
    import config

    benchmark = config.DEVICE_CFG.cudnn_benchmark
    torch.backends.cudnn.deterministic = not benchmark
    torch.backends.cudnn.benchmark = benchmark


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Cria (ou reutiliza) um logger configurado com saída em console e arquivo.

    Args:
        name: nome do logger (geralmente ``__name__``).
        log_file: caminho opcional para persistir os logs em disco.

    Returns:
        Instância de :class:`logging.Logger` pronta para uso.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # evita handlers duplicados em chamadas repetidas
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# I/O de imagem tolerante a caminhos não-ASCII
# --------------------------------------------------------------------------- #
# ``cv2.imread``/``cv2.imwrite`` repassam o caminho para a API ANSI do
# Windows e falham quando ele contém qualquer caractere fora do ASCII —
# sem exceção: o read devolve ``None`` e o write devolve ``False``.
#
# Isso não é hipotético para este projeto: ``config.BASE_DIR`` é sempre
# ABSOLUTO (``Path(__file__).resolve()``), então basta o projeto estar em
# "Área de Trabalho", "Documentos do João" ou qualquer pasta acentuada
# para que TODA leitura de imagem do pipeline devolva ``None``.
#
# As funções abaixo movem o acesso ao disco para o Python (que lida com
# Unicode corretamente) e deixam para o OpenCV apenas a decodificação dos
# bytes já em memória.

def ler_imagem(caminho: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Lê uma imagem do disco tolerando acentos no caminho.

    Args:
        caminho: arquivo a ler.
        flags: mesma semântica do parâmetro de ``cv2.imread``.

    Returns:
        A imagem decodificada, ou ``None`` se o arquivo não existir ou
        não for uma imagem válida — mesmo contrato de ``cv2.imread``.
    """
    try:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
    except OSError:
        return None
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, flags)


def escrever_imagem(caminho: Path, imagem: np.ndarray) -> None:
    """Grava uma imagem no disco tolerando acentos no caminho.

    A imagem é gravada como está, sem conversão de cor: quem tem um array
    RGB deve usar :func:`save_image`.

    Raises:
        RuntimeError: se o OpenCV não conseguir codificar a imagem.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(caminho.suffix or ".png", imagem)
    if not ok:
        raise RuntimeError(f"Falha ao codificar a imagem para {caminho}")
    buffer.tofile(str(caminho))


# --------------------------------------------------------------------------- #
# Visualização
# --------------------------------------------------------------------------- #

def mask_to_color(mask: np.ndarray, color: Sequence[int] = (255, 0, 0)) -> np.ndarray:
    """Converte uma máscara binária (H, W) em uma imagem colorida (H, W, 3).

    Args:
        mask: array binário 0/1 ou 0/255.
        color: cor RGB usada para os pixels de primeiro plano.

    Returns:
        Imagem RGB uint8 com a região da máscara pintada com ``color``.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    colored = np.zeros((*mask_bin.shape, 3), dtype=np.uint8)
    colored[mask_bin == 1] = color
    return colored


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Sobrepõe uma máscara semitransparente sobre a imagem original.

    Args:
        image: imagem RGB uint8 (H, W, 3).
        mask: máscara binária (H, W).
        color: cor RGB da sobreposição.
        alpha: opacidade da máscara (0 = invisível, 1 = opaca).

    Returns:
        Imagem RGB uint8 com a máscara sobreposta.
    """
    colored_mask = mask_to_color(mask, color)
    mask_bin = (mask > 0)
    overlay = image.copy()
    overlay[mask_bin] = (
        (1 - alpha) * image[mask_bin] + alpha * colored_mask[mask_bin]
    ).astype(np.uint8)
    return overlay


def draw_contour(
    image: np.ndarray, mask: np.ndarray, color: Sequence[int] = (0, 255, 0), thickness: int = 2
) -> np.ndarray:
    """Desenha o contorno externo da máscara sobre a imagem.

    Args:
        image: imagem RGB uint8.
        mask: máscara binária.
        color: cor do contorno.
        thickness: espessura da linha em pixels.

    Returns:
        Cópia da imagem com o contorno desenhado.
    """
    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = image.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def save_image(path: Path, image: np.ndarray) -> None:
    """Salva uma imagem RGB (uint8) em disco, criando diretórios se necessário.

    Args:
        path: caminho de destino.
        image: imagem RGB uint8.
    """
    escrever_imagem(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_figure(fig, out_path: Path, descricao: str, logger: Optional[logging.Logger] = None) -> None:
    """Salva uma figura matplotlib, fecha-a e registra o caminho.

    Fechar a figura é o que impede o vazamento de memória em rotinas que
    geram muitos gráficos em sequência — esquecê-lo é justamente o erro
    que centralizar este trecho evita.

    ``matplotlib`` é importado aqui dentro, e não no topo do módulo, para
    que :mod:`measure` continue importável com apenas ``cv2``/``numpy``:
    matplotlib pertence ao grupo de dependências de gráficos, não ao de
    medição.

    Args:
        fig: figura a salvar.
        out_path: caminho de destino do arquivo.
        descricao: texto usado na mensagem de log (ex.: "Curva ROC").
        logger: logger onde registrar; se ``None``, nada é registrado.
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("%s salva em %s", descricao, out_path)
