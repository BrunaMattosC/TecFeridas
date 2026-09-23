"""
config.py
=========
Configuração central do projeto de segmentação e medição de feridas.

Todas as constantes, hiperparâmetros e caminhos usados pelos demais
módulos vivem aqui para evitar "magic numbers" espalhados pelo código
(princípio DRY) e para permitir alterar o comportamento do pipeline
inteiro editando um único arquivo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"

OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

for _dir in (
    DATASET_DIR, RAW_DIR, IMAGES_DIR, MASKS_DIR,
    OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    REPORTS_DIR, FIGURES_DIR, PREDICTIONS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# Slug do dataset no Kaggle (usado pela Kaggle API):
#   https://www.kaggle.com/datasets/leoscode/wound-segmentation-images
KAGGLE_DATASET_SLUG = "leoscode/wound-segmentation-images"


# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #

SEED: int = 42


# --------------------------------------------------------------------------- #
# Leitura de variáveis de ambiente
# --------------------------------------------------------------------------- #
# Permitem mudar o comportamento de hardware sem editar o arquivo — útil
# para comparar CPU x GPU (``DEVICE=cpu python train.py``) sem tocar no
# código e sem risco de esquecer a alteração commitada.

def _env_bool(nome: str, padrao: bool) -> bool:
    """Lê um booleano de variável de ambiente ("1", "true", "sim", "yes")."""
    valor = os.environ.get(nome)
    if valor is None:
        return padrao
    return valor.strip().lower() in ("1", "true", "t", "sim", "s", "yes", "y", "on")


def _env_int(nome: str, padrao: int) -> int:
    """Lê um inteiro de variável de ambiente, ignorando valores inválidos."""
    try:
        return int(os.environ[nome])
    except (KeyError, ValueError):
        return padrao


# --------------------------------------------------------------------------- #
# Hardware / Dispositivo
# --------------------------------------------------------------------------- #

@dataclass
class DeviceConfig:
    """Configuração CENTRALIZADA de dispositivo (ver :mod:`hardware`).

    Todo o projeto — treino, validação, teste, inferência, geração de
    máscaras e avaliação — obtém seu ``torch.device`` a partir daqui, via
    :func:`hardware.resolver_dispositivo`. Não há nenhum outro lugar no
    código que decida entre GPU e CPU.
    """

    # "auto" = usa GPU se houver uma utilizável; "cpu" força CPU.
    preferred: str = os.environ.get("DEVICE", "auto")

    # Quando ``True``, o TREINO é abortado com erro explícito se a GPU não
    # estiver disponível, em vez de iniciar uma execução muito mais lenta
    # em CPU sem que ninguém perceba. As rotinas de inferência
    # (predict/batch_inference) continuam funcionando em CPU, mas sempre
    # avisam em voz alta quando existe GPU ociosa na máquina.
    # Desligue com: set REQUIRE_GPU=0   (Windows)  /  export REQUIRE_GPU=0
    require_gpu: bool = _env_bool("REQUIRE_GPU", True)

    # Índice da GPU em máquinas com mais de uma placa.
    gpu_index: int = _env_int("GPU_INDEX", 0)

    # Precisão mista: "fp16" (padrão, com GradScaler — é o que o código já
    # pressupunha), "bf16" (exige Ampere+; dispensa GradScaler e é mais
    # estável numericamente) ou "auto" (bf16 quando suportado).
    amp_dtype: str = os.environ.get("AMP_DTYPE", "fp16")

    # TF32 nas multiplicações de matriz/convolução (Ampere+). Acelera sem
    # o risco numérico do FP16, pois o acumulador continua em FP32.
    allow_tf32: bool = _env_bool("ALLOW_TF32", True)

    # cuDNN benchmark escolhe o algoritmo de convolução mais rápido para o
    # shape usado. DESLIGADO por padrão porque é incompatível com o
    # determinismo que ``utils.seed_everything`` estabelece — ligá-lo
    # mudaria a reprodutibilidade do projeto. Ligue com CUDNN_BENCHMARK=1
    # se aceitar trocar determinismo por velocidade.
    cudnn_benchmark: bool = _env_bool("CUDNN_BENCHMARK", False)

    # Log de memória/ocupação da GPU a cada N batches (0 = só por época).
    log_gpu_every_n_batches: int = _env_int("LOG_GPU_EVERY_N_BATCHES", 0)


DEVICE_CFG = DeviceConfig()


# --------------------------------------------------------------------------- #
# Dados / Pré-processamento
# --------------------------------------------------------------------------- #

def _num_workers_padrao() -> int:
    """Número de processos de carregamento adequado à máquina.

    O gargalo aqui é a augmentation da Albumentations em CPU (resize
    512x512 + CLAHE + blur), que precisa alimentar a GPU sem atrasá-la.
    Metade dos núcleos lógicos, limitada a 8, mantém CPU sobrando para o
    processo principal sem estourar a RAM com cópias do dataset.
    """
    nucleos = os.cpu_count() or 1
    return max(1, min(8, nucleos // 2))


@dataclass
class DataConfig:
    image_size: int = 512
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = _env_int("BATCH_SIZE", 8)
    num_workers: int = _env_int("NUM_WORKERS", _num_workers_padrao())
    # Memória paginada (pinned) no host: pré-requisito para que
    # ``tensor.to(device, non_blocking=True)`` copie de forma assíncrona,
    # sobrepondo transferência e computação. Ignorado quando em CPU.
    pin_memory: bool = True
    # Mantém os workers vivos entre épocas. No Windows, onde cada worker é
    # um processo novo (spawn), recriá-los a cada época custa segundos.
    persistent_workers: bool = True
    # Lotes que cada worker adianta. 2 é o padrão do PyTorch; 4 dá mais
    # folga para absorver variação no tempo de leitura em disco.
    prefetch_factor: int = 4
    # Descarta o último lote incompleto no TREINO (não em val/teste, para
    # não perder amostras de avaliação): mantém o shape constante, o que
    # evita realocação de memória na GPU a cada época.
    drop_last_train: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    # Uma das: "unet", "unetplusplus", "deeplabv3plus", "fpn"
    architecture: str = "unet"
    # Uma das: "resnet34", "efficientnet-b0", "efficientnet-b3"
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 1
    activation: str | None = None  # logits crus; sigmoid aplicado na loss/inferência


# --------------------------------------------------------------------------- #
# Treinamento
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-5
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    use_amp: bool = True
    grad_clip_norm: float | None = 1.0
    monitor_metric: str = "val_dice"  # métrica usada p/ early stopping / checkpoint
    # MANTIDO POR COMPATIBILIDADE. A decisão de dispositivo mudou de lugar:
    # agora vive em ``DEVICE_CFG`` e é resolvida por
    # :func:`hardware.resolver_dispositivo`, que verifica de fato se a GPU
    # está utilizável e avisa em vez de cair para CPU em silêncio.
    device: str = "cuda"


DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()


# --------------------------------------------------------------------------- #
# Medição (pós-processamento da máscara + conversão pixel -> unidades reais)
# --------------------------------------------------------------------------- #

@dataclass
class MeasurementConfig:
    """Parâmetros do pós-processamento aplicado à máscara antes de medir.

    Os valores padrão são conservadores: eliminam apenas ruído claramente
    espúrio (componentes minúsculos, buracos internos de poucos pixels) sem
    alterar significativamente a geometria da ferida segmentada.
    """

    # Um componente conectado só é considerado ferida se tiver, ao mesmo
    # tempo, área >= min_component_area_px e área >= min_component_area_ratio
    # da área total da imagem. Evita medir respingos de falso positivo.
    min_component_area_px: int = 50
    min_component_area_ratio: float = 0.0002  # 0,02% da imagem

    # Operações morfológicas são DESATIVADAS por padrão (0): elas
    # deslocam a borda da ferida e, portanto, enviesam justamente a área
    # e o perímetro que queremos medir (uma abertura 3x3 elíptica remove
    # os cantos de um retângulo). A remoção de ruído é feita pelo filtro
    # de área acima, que é exato e preserva a borda. Ative apenas se as
    # máscaras do seu modelo vierem com borda serrilhada/quebradiça.
    opening_kernel_size: int = 0  # remove ruído "sal" (pixels isolados)
    closing_kernel_size: int = 0  # une falhas finas na borda
    # Preenche buracos internos da máscara (ex.: tecido de granulação
    # classificado como fundo dentro da ferida). Não afeta a borda externa.
    fill_holes: bool = True

    # Conectividade usada em connectedComponents (4 ou 8).
    connectivity: int = 8
    # Número máximo de feridas reportadas por imagem (as maiores primeiro).
    # None = sem limite.
    max_wounds: int | None = 10


MEASURE_CFG = MeasurementConfig()

# Fator de escala padrão. Se ``None``, todas as medidas são reportadas
# apenas em pixels — o projeto NUNCA infere escala física sozinho.
DEFAULT_PIXELS_POR_CM: float | None = None
