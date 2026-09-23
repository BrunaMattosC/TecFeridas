"""
dataset.py
==========
Responsável por:

1. Baixar o dataset "Wound Segmentation Images" via API do Kaggle
   (sem intervenção manual), descompactar e organizar as pastas.
2. Exploração automática dos dados (contagens, resolução, distribuição
   de área das feridas, histogramas, exemplos aleatórios).
3. Definição do :class:`WoundDataset` (PyTorch ``Dataset``) e das
   transformações (Albumentations) usadas em treino/validação/teste.
4. Divisão determinística em treino (70%) / validação (15%) / teste (15%).

Os itens 1, 2 e 4 dependem apenas de ``cv2``/``numpy``/``pandas``; o item 3 e
a classe :class:`WoundDataset` exigem ``albumentations`` e ``torch``, que são
importados de forma tolerante (ver ``DEEP_LEARNING_DISPONIVEL``). Assim é
possível baixar e explorar o dataset em um ambiente mínimo.

Credenciais do Kaggle aceitas (qualquer uma das três):
    - ``KAGGLE_API_TOKEN`` (token moderno, formato ``KGAT_...``);
    - ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``;
    - arquivo ``~/.kaggle/kaggle.json``.

Referência do dataset:
    https://www.kaggle.com/datasets/leoscode/wound-segmentation-images
"""

from __future__ import annotations

import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import config
from utils import get_logger, ler_imagem

# --------------------------------------------------------------------------- #
# Import opcional da pilha de deep learning
# --------------------------------------------------------------------------- #
# O download, a organizacao das pastas, o pareamento imagem/mascara, a divisao
# treino/val/teste e a exploracao dependem apenas de cv2/numpy/pandas. Manter
# ``albumentations`` e ``torch`` como imports obrigatorios no topo do modulo
# obrigaria a instalar toda a pilha de deep learning so para baixar os dados.
# Importando-os de forma tolerante, ``python dataset.py`` passa a funcionar em
# um ambiente minimo, e apenas as funcoes que realmente precisam deles falham
# — com uma mensagem clara — quando nao estao instalados.
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import Dataset

    DEEP_LEARNING_DISPONIVEL = True
except ImportError:  # pragma: no cover - depende do ambiente
    A = None
    ToTensorV2 = None
    Dataset = object  # permite declarar WoundDataset sem torch instalado
    DEEP_LEARNING_DISPONIVEL = False


def _exigir_deep_learning(recurso: str) -> None:
    """Falha com mensagem acionavel se a pilha de DL nao estiver instalada."""
    if not DEEP_LEARNING_DISPONIVEL:
        raise RuntimeError(
            f"{recurso} exige 'albumentations' e 'torch', que nao estao instalados. "
            "Instale as dependencias completas com `pip install -r requirements.txt`. "
            "O download e a exploracao do dataset funcionam sem elas."
        )

logger = get_logger(__name__, config.LOGS_DIR / "dataset.log")

# Reexportado de :mod:`config` para não quebrar ``from dataset import
# IMAGE_EXTENSIONS``, já usado por outros módulos.
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS


# --------------------------------------------------------------------------- #
# Etapa 1 — Download automático via API do Kaggle
# --------------------------------------------------------------------------- #

def download_dataset(force: bool = False) -> Path:
    """Garante que o dataset esteja disponível localmente, baixando-o
    automaticamente da API do Kaggle caso ainda não exista.

    Requer credenciais do Kaggle configuradas por **um** destes meios:

    1. ``KAGGLE_API_TOKEN`` — token moderno (formato ``KGAT_...``), gerado
       em https://www.kaggle.com/settings. Exige ``kaggle >= 2.0``;
    2. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` — variáveis de ambiente;
    3. ``~/.kaggle/kaggle.json`` — arquivo padrão da API do Kaggle.

    Nunca grave credenciais dentro do projeto: prefira variáveis de
    ambiente ou o arquivo padrão, ambos fora do diretório versionado.

    Args:
        force: se ``True``, baixa novamente mesmo que os dados já existam.

    Returns:
        Caminho para a pasta onde os dados brutos foram extraídos.

    Raises:
        RuntimeError: se o pacote ``kaggle`` não estiver instalado ou as
            credenciais não estiverem configuradas.
    """
    already_present = config.RAW_DIR.exists() and any(config.RAW_DIR.iterdir())

    if already_present and not force:
        logger.info("Dataset já presente em %s — pulando download.", config.RAW_DIR)
        return config.RAW_DIR

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "O pacote 'kaggle' não está instalado. Instale com "
            "`pip install kaggle` e configure suas credenciais em "
            "~/.kaggle/kaggle.json antes de continuar."
        ) from exc

    logger.info("Autenticando na API do Kaggle...")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "Falha ao autenticar na API do Kaggle. Configure UMA destas opcoes:\n"
            "  1) export KAGGLE_API_TOKEN=KGAT_...   (token moderno; exige kaggle>=2.0)\n"
            "  2) export KAGGLE_USERNAME=... e export KAGGLE_KEY=...\n"
            "  3) salve o kaggle.json em ~/.kaggle/kaggle.json\n"
            "Gere as credenciais em https://www.kaggle.com/settings."
        ) from exc

    logger.info("Baixando dataset '%s'...", config.KAGGLE_DATASET_SLUG)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(
        config.KAGGLE_DATASET_SLUG, path=str(config.RAW_DIR), unzip=False, quiet=False
    )

    # Descompacta todos os .zip encontrados
    for zip_path in config.RAW_DIR.glob("*.zip"):
        logger.info("Descompactando %s...", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(config.RAW_DIR)
        zip_path.unlink()

    _organize_folders()
    logger.info("Dataset pronto em %s.", config.RAW_DIR)
    return config.RAW_DIR


def _organize_folders() -> None:
    """Localiza recursivamente imagens e máscaras dentro da pasta bruta
    do dataset e as organiza em ``dataset/images`` e ``dataset/masks``.

    O dataset original costuma vir com subpastas (ex.: ``images/`` e
    ``labels/`` ou ``masks/``); esta função tenta identificar os pares
    imagem/máscara pelo nome do arquivo.
    """
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.MASKS_DIR.mkdir(parents=True, exist_ok=True)

    all_files = [
        p for p in config.RAW_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]

    mask_keywords = ("mask", "label", "gt", "annotation")
    images: List[Path] = []
    masks: List[Path] = []

    for f in all_files:
        parts = [p.lower() for p in f.parts]
        is_mask = any(any(kw in part for kw in mask_keywords) for part in parts) or any(
            kw in f.stem.lower() for kw in mask_keywords
        )
        if is_mask:
            masks.append(f)
        else:
            images.append(f)

    if not images or not masks:
        logger.warning(
            "Não foi possível separar automaticamente imagens (%d) e "
            "máscaras (%d) — verifique manualmente a estrutura em %s.",
            len(images), len(masks), config.RAW_DIR,
        )

    for f in images:
        dest = config.IMAGES_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
    for f in masks:
        dest = config.MASKS_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    logger.info(
        "Organização concluída: %d imagens e %d máscaras copiadas.",
        len(list(config.IMAGES_DIR.glob('*'))), len(list(config.MASKS_DIR.glob('*'))),
    )


# --------------------------------------------------------------------------- #
# Pareamento imagem <-> máscara
# --------------------------------------------------------------------------- #

def contar_imagens(pasta: Path) -> int:
    """Conta apenas arquivos de imagem em uma pasta.

    ``Path.glob('*')`` inclui arquivos ocultos -- diferente do shell --, de
    modo que um ``.gitkeep`` era contado junto e inflava os totais
    reportados nos logs em uma unidade.
    """
    if not pasta.exists():
        return 0
    return sum(1 for p in pasta.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def list_image_mask_pairs(
    images_dir: Optional[Path] = None, masks_dir: Optional[Path] = None
) -> List[Tuple[Path, Path]]:
    """Constrói a lista de pares (imagem, máscara) casando pelo nome
    base do arquivo (ignorando extensão e sufixos comuns de máscara).

    Args:
        images_dir: pasta das imagens; ``None`` usa ``config.IMAGES_DIR``.
        masks_dir: pasta das máscaras; ``None`` usa ``config.MASKS_DIR``.

    Returns:
        Lista de tuplas ``(caminho_imagem, caminho_mascara)``.

    Raises:
        FileNotFoundError: se nenhuma imagem ou máscara for encontrada.
    """
    images_dir = images_dir or config.IMAGES_DIR
    masks_dir = masks_dir or config.MASKS_DIR

    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    masks = sorted([p for p in masks_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])

    if not images or not masks:
        raise FileNotFoundError(
            f"Nenhuma imagem/máscara encontrada em {images_dir} / {masks_dir}. "
            "Execute download_dataset() primeiro."
        )

    mask_lookup = {m.stem.lower().replace("_mask", "").replace("-mask", ""): m for m in masks}

    pairs: List[Tuple[Path, Path]] = []
    for img in images:
        key = img.stem.lower()
        mask = mask_lookup.get(key)
        if mask is None:
            # tentativa alternativa: mesmo índice posicional
            continue
        pairs.append((img, mask))

    if not pairs:
        # fallback: casa por ordem (assume mesma contagem/ordem alfabética)
        n = min(len(images), len(masks))
        pairs = list(zip(images[:n], masks[:n]))
        logger.warning(
            "Pareamento por nome falhou; usando pareamento posicional (%d pares).", n
        )

    return pairs


# --------------------------------------------------------------------------- #
# Divisão treino/validação/teste
# --------------------------------------------------------------------------- #

@dataclass
class DatasetSplits:
    train: List[Tuple[Path, Path]]
    val: List[Tuple[Path, Path]]
    test: List[Tuple[Path, Path]]


def split_dataset(
    pairs: Optional[List[Tuple[Path, Path]]] = None,
    seed: int = config.SEED,
    images_dir: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
) -> DatasetSplits:
    """Divide os pares imagem/máscara em treino (70%), validação (15%)
    e teste (15%) de forma determinística.

    Args:
        pairs: lista de pares a dividir; se ``None``, é obtida via
            :func:`list_image_mask_pairs`.
        seed: seed usada para embaralhar antes de dividir.
        images_dir: pasta das imagens quando ``pairs`` é ``None``;
            ``None`` usa ``config.IMAGES_DIR``.
        masks_dir: pasta das máscaras quando ``pairs`` é ``None``;
            ``None`` usa ``config.MASKS_DIR``.

    Returns:
        Instância de :class:`DatasetSplits` com as três partições.
    """
    if pairs is None:
        pairs = list_image_mask_pairs(images_dir, masks_dir)

    pairs = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * config.DATA_CFG.train_split)
    n_val = int(n * config.DATA_CFG.val_split)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    logger.info(
        "Split: %d treino / %d validação / %d teste (total=%d).",
        len(train_pairs), len(val_pairs), len(test_pairs), n,
    )
    return DatasetSplits(train=train_pairs, val=val_pairs, test=test_pairs)


# --------------------------------------------------------------------------- #
# Etapa 3 — Transformações (Albumentations)
# --------------------------------------------------------------------------- #

def get_train_transforms(image_size: int = config.DATA_CFG.image_size) -> A.Compose:
    """Pipeline de augmentation para treino.

    Inclui flips, rotação, ShiftScaleRotate, CLAHE, brilho/contraste
    aleatórios, blur gaussiano e variação de matiz/saturação/valor,
    seguido de normalização ImageNet e conversão para tensor.

    Raises:
        RuntimeError: se ``albumentations``/``torch`` não estiverem instalados.
    """
    _exigir_deep_learning("get_train_transforms()")
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=35, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.Normalize(mean=config.DATA_CFG.mean, std=config.DATA_CFG.std),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = config.DATA_CFG.image_size) -> A.Compose:
    """Pipeline determinístico (sem augmentation) para validação/teste/inferência.

    Raises:
        RuntimeError: se ``albumentations``/``torch`` não estiverem instalados.
    """
    _exigir_deep_learning("get_eval_transforms()")
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=config.DATA_CFG.mean, std=config.DATA_CFG.std),
            ToTensorV2(),
        ]
    )


# --------------------------------------------------------------------------- #
# Dataset PyTorch
# --------------------------------------------------------------------------- #

class WoundDataset(Dataset):
    """Dataset PyTorch para pares (imagem, máscara) de feridas.

    Args:
        pairs: lista de tuplas ``(caminho_imagem, caminho_mascara)``.
        transform: pipeline Albumentations aplicado a imagem e máscara.
    """

    def __init__(self, pairs: List[Tuple[Path, Path]], transform: Optional["A.Compose"] = None) -> None:
        _exigir_deep_learning("WoundDataset")
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        image = ler_imagem(img_path)
        if image is None:
            raise FileNotFoundError(f"Falha ao ler imagem: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = ler_imagem(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Falha ao ler máscara: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        return image, mask.float(), img_path.name


# --------------------------------------------------------------------------- #
# Etapa 2 — Exploração de dados
# --------------------------------------------------------------------------- #

def explore_dataset(pairs: Optional[List[Tuple[Path, Path]]] = None) -> pd.DataFrame:
    """Gera estatísticas exploratórias do dataset: contagens, resolução
    média e distribuição do tamanho (área) das feridas.

    Args:
        pairs: lista de pares a explorar; se ``None``, usa todos os
            pares encontrados no dataset organizado.

    Returns:
        DataFrame com uma linha por imagem contendo ``altura``,
        ``largura`` e ``area_ferida_pixels`` / ``percentual_area``.
    """
    if pairs is None:
        pairs = list_image_mask_pairs()

    rows = []
    for img_path, mask_path in pairs:
        mask = ler_imagem(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        area = int(np.sum(mask > 127))
        rows.append(
            {
                "arquivo": img_path.name,
                "altura": h,
                "largura": w,
                "area_ferida_pixels": area,
                "percentual_area": 100.0 * area / (h * w),
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Exploração: %d imagens, %d máscaras, resolução média %.0fx%.0f.",
        contar_imagens(config.IMAGES_DIR),
        contar_imagens(config.MASKS_DIR),
        df["altura"].mean() if not df.empty else 0,
        df["largura"].mean() if not df.empty else 0,
    )
    return df


if __name__ == "__main__":
    download_dataset()
    df = explore_dataset()
    df.to_csv(config.REPORTS_DIR / "exploracao_dataset.csv", index=False)
    logger.info("Relatório de exploração salvo em %s", config.REPORTS_DIR / "exploracao_dataset.csv")
