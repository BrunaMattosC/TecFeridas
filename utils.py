"""
utils.py
========
Funções utilitárias compartilhadas por todos os scripts do projeto:

- fixação de seeds para reprodutibilidade;
- configuração de logging;
- funções de visualização (overlay de máscara, grids de exemplos);
- helpers genéricos de I/O.

``torch`` é importado sob demanda (dentro de :func:`seed_everything` e
:func:`get_device`) para que as rotinas puramente de imagem — usadas
pela medição em :mod:`measure` — possam ser importadas e testadas sem
carregar toda a pilha de deep learning.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:  # apenas para type checking — não importa torch em runtime
    import torch


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
# Device
# --------------------------------------------------------------------------- #

def get_device(
    preferred: str = "cuda",
    exigir_gpu: Optional[bool] = None,
    silencioso: bool = False,
) -> "torch.device":
    """Retorna o device a ser usado, delegando a :mod:`hardware`.

    Esta função é apenas um atalho compatível com o código já existente:
    a decisão real (e a verificação de que a GPU está mesmo utilizável)
    vive em :func:`hardware.resolver_dispositivo`, para que exista um
    único ponto de decisão no projeto.

    Diferente da versão anterior, **não há fallback silencioso**: se a
    máquina tem GPU e o PyTorch não consegue usá-la, um aviso explicando o
    motivo é impresso; com ``REQUIRE_GPU`` ligado, o erro é levantado.

    Args:
        preferred: device desejado ("cuda"/"auto" ou "cpu").
        exigir_gpu: sobrescreve ``config.DEVICE_CFG.require_gpu``. Use
            ``False`` em rotinas de inferência, que continuam úteis em CPU.
        silencioso: suprime o aviso de GPU ociosa.

    Returns:
        Instância de :class:`torch.device`.

    Raises:
        hardware.GPUIndisponivelError: se a GPU for exigida e não estiver
            disponível.
    """
    from hardware import resolver_dispositivo

    return resolver_dispositivo(
        preferido=preferred, exigir_gpu=exigir_gpu, silencioso=silencioso
    )


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


# --------------------------------------------------------------------------- #
# I/O de imagem tolerante a caminhos não-ASCII
# --------------------------------------------------------------------------- #
# ``cv2.imread`` e ``cv2.imwrite`` repassam o caminho para a API ANSI do
# Windows e falham com qualquer caractere fora do ASCII — em silêncio: o
# read devolve ``None`` e o write devolve ``False``, sem exceção.
#
# Isso não é hipotético para este projeto. ``config.BASE_DIR`` é sempre
# ABSOLUTO (``Path(__file__).resolve()``), então basta o projeto estar em
# "Área de Trabalho", "Documentos do João" ou qualquer pasta acentuada para
# que TODA leitura de imagem do pipeline devolva ``None`` — treino,
# inferência, avaliação e exploração deixam de funcionar de uma vez.
#
# A solução move o acesso ao disco para o Python, que trata Unicode
# corretamente, e deixa para o OpenCV apenas decodificar os bytes já em
# memória. O comportamento em caminhos ASCII é idêntico ao de antes.

def ler_imagem(caminho: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Lê uma imagem do disco tolerando acentos no caminho.

    Substitui ``cv2.imread`` com o mesmo contrato.

    Args:
        caminho: arquivo a ler.
        flags: mesma semântica do parâmetro de ``cv2.imread``
            (ex.: ``cv2.IMREAD_GRAYSCALE``).

    Returns:
        A imagem decodificada, ou ``None`` se o arquivo não existir, não
        puder ser lido ou não for uma imagem válida.
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

    Substitui ``cv2.imwrite``. A imagem é gravada como está, sem conversão
    de cor: quem tem um array RGB deve usar :func:`save_image`.

    Args:
        caminho: destino; os diretórios são criados se necessário.
        imagem: array a gravar, no formato que o OpenCV espera (BGR para
            imagens coloridas).

    Raises:
        RuntimeError: se o OpenCV não conseguir codificar a imagem.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(caminho.suffix or ".png", imagem)
    if not ok:
        raise RuntimeError(f"Falha ao codificar a imagem para {caminho}")
    buffer.tofile(str(caminho))
