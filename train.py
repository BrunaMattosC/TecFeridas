"""
train.py
========
Etapa 7 — Treinamento do modelo de segmentação.

Recursos implementados:
    - Mixed Precision (AMP) quando rodando em GPU;
    - ReduceLROnPlateau (baseado na métrica de validação monitorada);
    - Early Stopping;
    - Checkpoint do melhor modelo;
    - Logging via TensorBoard;
    - Histórico completo de métricas salvo em CSV.

Uso:
    python train.py
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import WoundDataset, get_eval_transforms, get_train_transforms, split_dataset
from hardware import (
    AmostradorGPU,
    ajustar_backends,
    banner_hardware,
    conferir_lote_no_dispositivo,
    conferir_modelo_no_dispositivo,
    configuracao_amp,
    contexto_autocast,
    criar_grad_scaler,
    memoria_gpu,
    obter_hardware,
    para_dispositivo,
    parametros_dataloader,
    resolver_dispositivo,
    texto_memoria_gpu,
    verificar_uso_real_da_gpu,
)
from losses import DiceBCELoss
from metrics import SegmentationMetrics, dice_coefficient
from models import build_model, count_parameters
from utils import get_logger, seed_everything

logger = get_logger(__name__, config.LOGS_DIR / "train.log")

_SEP = "=" * 66


# --------------------------------------------------------------------------- #
# Compatibilidade de AMP entre versoes do PyTorch
# --------------------------------------------------------------------------- #
# A escolha de API (``torch.amp`` novo x ``torch.cuda.amp`` depreciado) e a
# decisao de dtype (fp16/bf16) foram movidas para :mod:`hardware`, para que
# treino, validacao, avaliacao e inferencia usem exatamente a mesma regra.
# Os nomes abaixo permanecem como atalhos compativeis com o codigo anterior.

def autocast_context(enabled: bool, device: torch.device | None = None, dtype=None):
    """Devolve o contexto de autocast compativel com esta versao do torch."""
    return contexto_autocast(device or torch.device("cuda"), enabled, dtype)


class EarlyStopping:
    """Interrompe o treino quando a métrica monitorada para de melhorar.

    Args:
        patience: número de épocas sem melhora antes de parar.
        mode: "max" se a métrica deve aumentar, "min" se deve diminuir.
    """

    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Atualiza o estado com o novo score. Retorna ``True`` se houve melhora."""
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def build_dataloaders(device: torch.device | None = None) -> Dict[str, DataLoader]:
    """Cria os DataLoaders de treino, validação e teste a partir do split do dataset.

    O split (70/15/15, seed fixa) é exatamente o mesmo de antes. O que muda
    conforme o dispositivo são apenas parâmetros de *transporte* dos dados
    — ``pin_memory``, ``persistent_workers``, ``prefetch_factor`` —, que
    não alteram quais amostras chegam ao modelo nem em que ordem.

    Args:
        device: dispositivo de destino; se ``None``, é resolvido via
            :func:`hardware.resolver_dispositivo`.
    """
    device = device or resolver_dispositivo()
    splits = split_dataset()

    train_ds = WoundDataset(splits.train, transform=get_train_transforms())
    val_ds = WoundDataset(splits.val, transform=get_eval_transforms())
    test_ds = WoundDataset(splits.test, transform=get_eval_transforms())

    params_treino = parametros_dataloader(device, treino=True)
    params_eval = parametros_dataloader(device, treino=False)

    logger.info(
        "DataLoader: batch_size=%d num_workers=%d pin_memory=%s "
        "persistent_workers=%s prefetch_factor=%s drop_last(treino)=%s",
        params_treino["batch_size"], params_treino["num_workers"],
        params_treino["pin_memory"], params_treino.get("persistent_workers", False),
        params_treino.get("prefetch_factor", "n/a"), params_treino["drop_last"],
    )

    return {
        "train": DataLoader(train_ds, shuffle=True, **params_treino),
        "val": DataLoader(val_ds, shuffle=False, **params_eval),
        "test": DataLoader(test_ds, shuffle=False, **params_eval),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    metrics: SegmentationMetrics,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    use_amp: bool = False,
    grad_clip_norm: float | None = None,
    amp_dtype: "torch.dtype | None" = None,
) -> Dict[str, float]:
    """Executa uma época de treino (se ``optimizer`` for informado) ou de avaliação.

    Modelo, imagens, máscaras, loss e métricas ficam TODOS em ``device``:
    nada é trazido de volta para a CPU dentro do laço, exceto os escalares
    finais de loss/dice (via ``.item()``), que são apenas números.

    Returns:
        Dicionário com a loss média, as métricas agregadas da época e os
        tempos medidos (``tempo_total``, ``tempo_por_batch``).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    metrics.reset()

    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0
    log_a_cada = config.DEVICE_CFG.log_gpu_every_n_batches

    inicio = time.perf_counter()
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks, _ in tqdm(loader, desc="treino" if is_train else "avaliação", leave=False):
            # Transferência assíncrona host->device: só é de fato assíncrona
            # porque o DataLoader entrega os tensores em memória pinada.
            images = para_dispositivo(images, device)
            masks = para_dispositivo(masks, device)

            # Primeiro lote: prova explícita de que dados e modelo estão no
            # MESMO dispositivo, em vez de descobrir isso num RuntimeError
            # obscuro no meio da época.
            if n_batches == 0:
                conferir_lote_no_dispositivo(device, imagens=images, mascaras=masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # A FORWARD do modelo roda em precisão mista (é onde está o ganho:
            # convoluções nos Tensor Cores). A LOSS é calculada fora do
            # autocast, em float32, de propósito:
            #
            #   DiceLoss soma as probabilidades de todos os pixels da imagem
            #   (512*512 = 262.144). Em fp16, cujo maior valor finito é
            #   65.504, essa soma satura em +inf assim que ~25% da imagem tem
            #   probabilidade alta — e a loss vira NaN. Promovendo os logits a
            #   float32 antes da loss, o valor calculado é o MESMO que o do
            #   treino em precisão simples, e o gradiente continua fluindo de
            #   volta para os pesos em fp16 normalmente.
            with autocast_context(use_amp, device=device, dtype=amp_dtype):
                logits = model(images)

            logits = logits.float()
            loss = criterion(logits, masks)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Caminho usado em CPU e em AMP bf16 (que dispensa
                    # escalonamento por ter o mesmo expoente do fp32).
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            # As métricas do TorchMetrics vivem no mesmo device do modelo,
            # então nada volta para a CPU aqui — só os escalares no fim.
            logits_metricas = logits.detach()
            metrics.update(logits_metricas, masks)
            running_loss += loss.item()
            running_dice += dice_coefficient(logits_metricas, masks)
            n_batches += 1

            if log_a_cada and n_batches % log_a_cada == 0 and device.type == "cuda":
                logger.info("  batch %d | GPU: %s", n_batches, texto_memoria_gpu(device))

    # Sem sincronizar, o tempo medido terminaria antes da GPU concluir a
    # fila de kernels — e o treino pareceria mais rápido do que é.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decorrido = time.perf_counter() - inicio

    result = metrics.compute()
    result["loss"] = running_loss / max(n_batches, 1)
    result["dice"] = running_dice / max(n_batches, 1)
    result["tempo_total"] = decorrido
    result["tempo_por_batch"] = decorrido / max(n_batches, 1)
    return result


def _configurar_saida_padrao() -> None:
    """Volta o stdout para buffer de linha.

    Os banners e o relatório por época saem em stdout; os logs, em stderr.
    Com a saída redirecionada para um arquivo, o stdout vira buffer de
    bloco e o banner apareceria DEPOIS dos logs, fora de ordem. Buffer de
    linha mantém o relatório legível em `python train.py > treino.log 2>&1`.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # stdout substituído (ex.: pytest)
        pass


def _resolver_configuracoes(
    architecture: str | None, encoder_name: str | None, epochs: int | None
) -> tuple[config.ModelConfig, config.TrainConfig]:
    """Copia as configurações globais e aplica as sobrescritas recebidas.

    A cópia profunda evita que uma chamada com parâmetros personalizados
    contamine ``config.MODEL_CFG``/``config.TRAIN_CFG`` para as seguintes.
    """
    model_cfg = copy.deepcopy(config.MODEL_CFG)
    train_cfg = copy.deepcopy(config.TRAIN_CFG)
    if architecture:
        model_cfg.architecture = architecture
    if encoder_name:
        model_cfg.encoder_name = encoder_name
    if epochs:
        train_cfg.epochs = epochs
    return model_cfg, train_cfg


def _preparar_dispositivo(
    model_cfg: config.ModelConfig, train_cfg: config.TrainConfig
) -> tuple[torch.device, bool, "torch.dtype | None"]:
    """Resolve o dispositivo, ajusta backends, imprime o cabeçalho e confere a GPU.

    O dispositivo é decidido em um único lugar, com aviso alto se a GPU
    existir mas não estiver utilizável (e erro, se ``REQUIRE_GPU``). Antes
    de qualquer época começar, a GPU executa uma operação de verdade.

    Returns:
        Tupla ``(device, usar_amp, amp_dtype)``.

    Raises:
        RuntimeError: se o device for CUDA mas a verificação real falhar.
    """
    device = resolver_dispositivo()
    info = obter_hardware()
    ajustar_backends(device)

    usar_amp, amp_dtype, motivo_amp = configuracao_amp(device)

    print(banner_hardware(device, info))
    print()
    print(_SEP)
    print("INICIANDO TREINAMENTO")
    print(_SEP)
    print()
    print("Framework: %s %s" % (info.framework, info.versao_framework))
    print("Dispositivo selecionado: %s" % ("GPU" if device.type != "cpu" else "CPU"))
    print("Mixed precision: %s (%s)" % ("SIM" if usar_amp else "NAO", motivo_amp))
    print("Arquitetura: %s + %s" % (model_cfg.architecture, model_cfg.encoder_name))
    print("Batch size: %d" % config.DATA_CFG.batch_size)
    print("Epochs: %d" % train_cfg.epochs)
    print()

    if device.type == "cuda":
        if not verificar_uso_real_da_gpu(device):
            raise RuntimeError(
                "O dispositivo selecionado é CUDA, mas a GPU não conseguiu "
                "executar a operação de verificação. O treino foi abortado "
                "para não rodar silenciosamente em um estado inconsistente."
            )
        torch.cuda.reset_peak_memory_stats(device)
    print(_SEP)
    print()

    return device, usar_amp, amp_dtype


def _relatar_epoca(
    epoch: int,
    total_epocas: int,
    rotulo_device: str,
    device: torch.device,
    train_result: Dict[str, float],
    val_result: Dict[str, float],
    elapsed: float,
    uso_gpu: str,
    mem: Dict[str, float],
    ocupacao_media: float | None,
) -> None:
    """Relata a época para quem acompanha: log (stderr) e relatório (stdout)."""
    logger.info(
        "Época %02d/%d | %.1fs | train_loss=%.4f train_dice=%.4f | "
        "val_loss=%.4f val_dice=%.4f val_iou=%.4f",
        epoch, total_epocas, elapsed,
        train_result["loss"], train_result["dice"],
        val_result["loss"], val_result["dice"], val_result["iou"],
    )
    logger.info(
        "         Device: %s | GPU Usage: %s | GPU Memory: %s | "
        "treino %.1fs (%.3fs/batch) | validação %.1fs (%.3fs/batch)",
        rotulo_device, uso_gpu,
        texto_memoria_gpu(device) if mem else "n/a (CPU)",
        train_result["tempo_total"], train_result["tempo_por_batch"],
        val_result["tempo_total"], val_result["tempo_por_batch"],
    )

    print("Train Loss: %.5f" % train_result["loss"])
    print("Val Loss: %.5f" % val_result["loss"])
    print("Val Dice: %.5f" % val_result["dice"])
    if mem:
        print("GPU Memory: %.0f MB alocada / %.0f MB reservada / pico %.0f MB"
              % (mem["alocada"], mem["reservada"], mem["pico_alocado"]))
        print("GPU Usage: %s" % uso_gpu)
    print("Time: %.1fs (%.3fs/batch no treino)"
          % (elapsed, train_result["tempo_por_batch"]))
    print()

    # Alerta explícito de fallback: se em algum momento o modelo deixar
    # de estar na GPU esperada, isso aparece aqui e não meses depois.
    if device.type == "cuda" and ocupacao_media is not None and ocupacao_media < 5:
        logger.warning(
            "ATENÇÃO: a GPU foi detectada, porém sua ocupação média nesta "
            "época foi de apenas %.0f%%. Verifique se o carregamento de "
            "dados (num_workers) não está sendo o gargalo.", ocupacao_media
        )


def _registrar_epoca(
    writer: SummaryWriter,
    epoch: int,
    rotulo_device: str,
    optimizer: torch.optim.Optimizer,
    train_result: Dict[str, float],
    val_result: Dict[str, float],
    elapsed: float,
    mem: Dict[str, float],
    ocupacao_media: float | None,
) -> Dict[str, float]:
    """Grava as métricas da época no TensorBoard e devolve a linha do histórico.

    ``device`` e os tempos ficam no CSV de propósito: é o que permite
    comparar depois, objetivamente, o custo por época em CPU x GPU.
    """
    for split_name, result in (("train", train_result), ("val", val_result)):
        for metric_name, value in result.items():
            writer.add_scalar(f"{metric_name}/{split_name}", value, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("tempo/epoca_segundos", elapsed, epoch)
    if mem:
        writer.add_scalar("gpu/memoria_alocada_mb", mem["alocada"], epoch)
        writer.add_scalar("gpu/memoria_reservada_mb", mem["reservada"], epoch)
        if ocupacao_media is not None:
            writer.add_scalar("gpu/utilizacao_percent", ocupacao_media, epoch)

    row = {"epoch": epoch, "device": rotulo_device, "tempo_epoca_s": elapsed}
    row.update({f"train_{k}": v for k, v in train_result.items()})
    row.update({f"val_{k}": v for k, v in val_result.items()})
    if mem:
        row["gpu_memoria_alocada_mb"] = mem["alocada"]
        row["gpu_memoria_reservada_mb"] = mem["reservada"]
        row["gpu_memoria_pico_mb"] = mem["pico_alocado"]
        row["gpu_utilizacao_media_percent"] = ocupacao_media
    return row


def _salvar_checkpoint(
    caminho: Path,
    model: torch.nn.Module,
    model_cfg: config.ModelConfig,
    epoch: int,
    val_dice: float,
) -> None:
    """Salva os pesos do melhor modelo junto da configuração que o gerou."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg.__dict__,
            "epoch": epoch,
            "val_dice": val_dice,
        },
        caminho,
    )
    logger.info("Novo melhor modelo salvo (val_dice=%.4f) em %s", val_dice, caminho)


def train_model(
    architecture: str | None = None,
    encoder_name: str | None = None,
    epochs: int | None = None,
) -> Path:
    """Executa o pipeline completo de treinamento e retorna o caminho do melhor checkpoint.

    Args:
        architecture: sobrescreve ``config.MODEL_CFG.architecture`` se informado.
        encoder_name: sobrescreve ``config.MODEL_CFG.encoder_name`` se informado.
        epochs: sobrescreve ``config.TRAIN_CFG.epochs`` se informado.

    Returns:
        Caminho do arquivo ``.pt`` contendo os pesos do melhor modelo.
    """
    _configurar_saida_padrao()
    seed_everything(config.SEED)

    model_cfg, train_cfg = _resolver_configuracoes(architecture, encoder_name, epochs)
    device, usar_amp, amp_dtype = _preparar_dispositivo(model_cfg, train_cfg)

    loaders = build_dataloaders(device)

    # Modelo -> dispositivo, com conferência de que TODOS os parâmetros e
    # buffers foram de fato movidos.
    model = build_model(model_cfg).to(device)
    conferir_modelo_no_dispositivo(model, device, nome="modelo de segmentação")
    logger.info("Modelo em %s (%d parâmetros treináveis)",
                next(model.parameters()).device, count_parameters(model))
    if device.type == "cuda":
        logger.info("GPU após carregar o modelo: %s", texto_memoria_gpu(device))

    criterion = DiceBCELoss(train_cfg.dice_weight, train_cfg.bce_weight).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=train_cfg.lr_scheduler_factor, patience=train_cfg.lr_scheduler_patience
    )
    scaler = criar_grad_scaler(device, usar_amp, amp_dtype)
    early_stopping = EarlyStopping(patience=train_cfg.early_stopping_patience, mode="max")

    train_metrics = SegmentationMetrics(device)
    val_metrics = SegmentationMetrics(device)

    run_name = f"{model_cfg.architecture}_{model_cfg.encoder_name}_{int(time.time())}"
    writer = SummaryWriter(log_dir=str(config.LOGS_DIR / "tensorboard" / run_name))
    best_ckpt_path = config.CHECKPOINTS_DIR / f"best_{run_name}.pt"

    history: List[Dict[str, float]] = []
    best_val_dice = -1.0

    rotulo_device = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )

    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.time()
        print("Epoch %d/%d" % (epoch, train_cfg.epochs))
        print("Device: %s" % rotulo_device)
        if device.type == "cuda":
            m = memoria_gpu(device)
            print("GPU Memory Allocated: %.0f MB" % m["alocada"])
            print("GPU Memory Reserved: %.0f MB" % m["reservada"])
        print("Training...")

        # A ocupação média do núcleo gráfico durante a época é a evidência
        # mais direta de que a GPU está de fato executando o treino.
        with AmostradorGPU(device) as amostrador:
            train_result = run_epoch(
                model, loaders["train"], criterion, device, train_metrics,
                optimizer=optimizer, scaler=scaler, use_amp=usar_amp,
                grad_clip_norm=train_cfg.grad_clip_norm, amp_dtype=amp_dtype,
            )
        val_result = run_epoch(
            model, loaders["val"], criterion, device, val_metrics,
            use_amp=usar_amp, amp_dtype=amp_dtype,
        )

        scheduler.step(val_result["dice"])

        elapsed = time.time() - t0
        uso_gpu = amostrador.resumo()
        mem = memoria_gpu(device)

        _relatar_epoca(
            epoch, train_cfg.epochs, rotulo_device, device,
            train_result, val_result, elapsed, uso_gpu, mem, amostrador.media,
        )

        history.append(_registrar_epoca(
            writer, epoch, rotulo_device, optimizer,
            train_result, val_result, elapsed, mem, amostrador.media,
        ))

        improved = early_stopping.step(val_result["dice"])
        if improved:
            best_val_dice = val_result["dice"]
            _salvar_checkpoint(best_ckpt_path, model, model_cfg, epoch, best_val_dice)

        if early_stopping.should_stop:
            logger.info("Early stopping acionado na época %d.", epoch)
            break

    writer.close()
    history_df = pd.DataFrame(history)
    history_path = config.REPORTS_DIR / f"training_history_{run_name}.csv"
    history_df.to_csv(history_path, index=False)
    logger.info("Histórico de treinamento salvo em %s", history_path)

    return best_ckpt_path


if __name__ == "__main__":
    from hardware import GPUIndisponivelError

    try:
        best_path = train_model()
    except GPUIndisponivelError as exc:
        # Falha explícita e imediata em vez de horas de treino em CPU que
        # ninguém pediu (item 9 do escopo).
        print()
        print(_SEP)
        print("TREINAMENTO ABORTADO: GPU EXIGIDA E INDISPONÍVEL")
        print(_SEP)
        print(exc)
        print(_SEP)
        sys.exit(2)

    logger.info("Treinamento concluído. Melhor checkpoint: %s", best_path)
