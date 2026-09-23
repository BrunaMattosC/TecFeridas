"""
teccurar_pipeline.py
=====================
Camada de inferência do TecCurar: da fotografia ao conjunto de medidas.

Concentra o caminho "imagem -> máscara -> medidas -> escala" para que a
interface não precise conhecer modelo, dispositivo nem pós-processamento.
Reaproveita o que o projeto já tem — :mod:`predict` para segmentar,
:mod:`measure` para medir, :mod:`hardware` para resolver o dispositivo — em
vez de reimplementar qualquer uma dessas etapas.

    interface  ->  teccurar_pipeline  ->  predict / measure / hardware

Sobre a conversão para centímetros
----------------------------------
O pipeline **nunca infere escala sozinho**. Uma fotografia de ferida não
carrega informação de tamanho físico: a mesma ferida ocupa mais ou menos
pixels conforme a distância da câmera. A conversão para cm só acontece
quando o usuário fornece uma referência — um fator pixels/cm conhecido ou
um objeto de tamanho conhecido visível na imagem.

Sem referência, as medidas saem apenas em pixels e o status da calibração
diz isso explicitamente. A calibração por moedas obtida em
:mod:`calibrar_moedas` vale para o arranjo fotográfico daquelas fotos, e
aplicá-la a uma foto qualquer seria inventar um número.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from measure import (
    PixelScale,
    WoundMeasurements,
    annotate_measurements,
    clean_mask,
    format_measurements_text,
    measure_wounds,
    scale_from_options,
)
from utils import get_logger, ler_imagem, overlay_mask_on_image

logger = get_logger(__name__, config.LOGS_DIR / "teccurar_pipeline.log")

EXTENSOES_ACEITAS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CALIBRACAO_AUSENTE = "sem referencia fisica"
CALIBRACAO_MANUAL = "fator informado pelo usuario"
CALIBRACAO_REFERENCIA = "objeto de referencia na imagem"


# --------------------------------------------------------------------------- #
# Erros com mensagem destinada ao usuário final
# --------------------------------------------------------------------------- #

class ErroTecCurar(Exception):
    """Erro cuja mensagem pode ser exibida diretamente na interface."""


class ModeloIndisponivelError(ErroTecCurar):
    """Não há checkpoint treinado para segmentar."""


class ImagemInvalidaError(ErroTecCurar):
    """O arquivo escolhido não é uma imagem legível."""


class SegmentacaoVaziaError(ErroTecCurar):
    """O modelo rodou, mas não encontrou ferida na imagem."""


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #

@dataclass
class ResultadoAnalise:
    """Tudo que a interface e o relatório precisam de uma análise."""

    caminho_imagem: Path
    imagem_rgb: np.ndarray
    mascara: np.ndarray                 # binária 0/1, resolução da imagem
    sobreposicao: np.ndarray            # fotografia + máscara
    anotada: np.ndarray                 # fotografia + geometria e medidas
    feridas: List[WoundMeasurements]
    escala: Optional[PixelScale]
    status_calibracao: str
    checkpoint: Path
    quando: datetime = field(default_factory=datetime.now)

    @property
    def tem_centimetros(self) -> bool:
        """Se as medidas físicas puderam ser determinadas com segurança."""
        return self.escala is not None

    @property
    def principal(self) -> WoundMeasurements:
        """A maior ferida encontrada."""
        return self.feridas[0]

    def texto_medidas(self) -> str:
        """Resumo textual das medidas, no formato já usado pelo projeto."""
        return format_measurements_text(self.feridas)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

def localizar_checkpoint(preferido: Optional[Path] = None) -> Path:
    """Devolve o checkpoint a usar, ou explica por que não há um.

    Args:
        preferido: caminho escolhido explicitamente; se ``None``, procura o
            mais recente em ``checkpoints/``.

    Raises:
        ModeloIndisponivelError: com mensagem pronta para a interface.
    """
    if preferido is not None:
        if not Path(preferido).exists():
            raise ModeloIndisponivelError(
                "O arquivo de modelo indicado não foi encontrado:\n%s" % preferido
            )
        return Path(preferido)

    candidatos = sorted(config.CHECKPOINTS_DIR.glob("*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidatos:
        raise ModeloIndisponivelError(
            "Nenhum modelo treinado foi encontrado em 'checkpoints'.\n\n"
            "A segmentação automática depende de um modelo já treinado. "
            "Para gerar um, execute o treinamento do projeto (python train.py). "
            "Enquanto não houver modelo, a análise de novas fotografias não "
            "pode ser realizada."
        )
    return candidatos[0]


class _Segmentador:
    """Mantém o modelo carregado entre análises.

    Recarregar o checkpoint a cada imagem custaria segundos e é
    desnecessário: o modelo não muda entre uma análise e outra.
    """

    def __init__(self) -> None:
        self._caminho: Optional[Path] = None
        self._modelo = None
        self._device = None

    def preparar(self, checkpoint: Path):
        """Carrega o modelo se ainda não estiver carregado."""
        if self._modelo is not None and self._caminho == checkpoint:
            return self._modelo, self._device

        from hardware import ajustar_backends, resolver_dispositivo
        from predict import load_model_from_checkpoint

        # Inferência de uma imagem é útil em CPU; a GPU não é exigida.
        device = resolver_dispositivo(exigir_gpu=False)
        ajustar_backends(device)
        try:
            modelo = load_model_from_checkpoint(checkpoint, device)
        except Exception as exc:
            raise ModeloIndisponivelError(
                "O modelo em '%s' não pôde ser carregado.\n\n"
                "O arquivo pode estar corrompido ou ter sido gerado por uma "
                "versão incompatível do projeto.\n\nDetalhe técnico: %s"
                % (checkpoint.name, exc)
            ) from exc

        self._caminho, self._modelo, self._device = checkpoint, modelo, device
        logger.info("Modelo carregado de %s em %s", checkpoint.name, device)
        return modelo, device


_segmentador = _Segmentador()


# --------------------------------------------------------------------------- #
# Escala
# --------------------------------------------------------------------------- #

def montar_escala(
    pixels_por_cm: Optional[float] = None,
    referencia_pixels: Optional[float] = None,
    referencia_cm: Optional[float] = None,
) -> Tuple[Optional[PixelScale], str]:
    """Constrói a escala a partir do que o usuário informou.

    Returns:
        ``(escala, status)``. A escala é ``None`` quando nenhuma referência
        utilizável foi fornecida — e nesse caso nenhum número é inventado.
    """
    if pixels_por_cm:
        return scale_from_options(pixels_por_cm=pixels_por_cm), CALIBRACAO_MANUAL
    if referencia_pixels and referencia_cm:
        return (scale_from_options(referencia_pixels=referencia_pixels,
                                   referencia_cm=referencia_cm),
                CALIBRACAO_REFERENCIA)
    return None, CALIBRACAO_AUSENTE


# --------------------------------------------------------------------------- #
# Análise
# --------------------------------------------------------------------------- #

def analisar_imagem(
    caminho: Path,
    checkpoint: Optional[Path] = None,
    escala: Optional[PixelScale] = None,
    status_calibracao: str = CALIBRACAO_AUSENTE,
    threshold: float = 0.5,
) -> ResultadoAnalise:
    """Executa o fluxo completo sobre uma fotografia.

    Segmenta, limpa a máscara, mede cada ferida e monta as visualizações.

    Args:
        caminho: fotografia a analisar.
        checkpoint: modelo a usar; ``None`` usa o mais recente.
        escala: conversão pixel->cm; ``None`` mantém tudo em pixels.
        status_calibracao: como a escala foi obtida, para o relatório.
        threshold: limiar de binarização da probabilidade.

    Raises:
        ImagemInvalidaError, ModeloIndisponivelError, SegmentacaoVaziaError.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise ImagemInvalidaError("O arquivo não foi encontrado:\n%s" % caminho)
    if caminho.suffix.lower() not in EXTENSOES_ACEITAS:
        raise ImagemInvalidaError(
            "Formato não suportado: %s\n\nUse JPG, JPEG ou PNG."
            % (caminho.suffix or "sem extensão")
        )
    if ler_imagem(caminho) is None:
        raise ImagemInvalidaError(
            "O arquivo não pôde ser lido como imagem.\n\n"
            "Ele pode estar corrompido ou não ser realmente uma fotografia."
        )

    caminho_modelo = localizar_checkpoint(checkpoint)
    modelo, device = _segmentador.preparar(caminho_modelo)

    from predict import predict_mask

    try:
        imagem_rgb, mascara_crua = predict_mask(modelo, caminho, device, threshold=threshold)
    except Exception as exc:
        raise ErroTecCurar(
            "A segmentação não pôde ser concluída para esta imagem.\n\n"
            "Detalhe técnico: %s" % exc
        ) from exc

    # A máscara medida é também a exibida, para que números e imagem nunca
    # discordem — mesma regra já usada em predict.predict_single_image.
    mascara = clean_mask(mascara_crua)
    feridas = measure_wounds(mascara, pixels_por_cm=escala, cleanup=False)
    if not feridas:
        raise SegmentacaoVaziaError(
            "Nenhuma ferida foi identificada nesta fotografia.\n\n"
            "Verifique se a imagem mostra a lesão com nitidez e boa "
            "iluminação, e se o enquadramento inclui a região afetada."
        )

    sobreposicao = overlay_mask_on_image(imagem_rgb, mascara, color=(40, 200, 90))
    anotada = annotate_measurements(imagem_rgb, feridas, mask=mascara)

    logger.info("Analise de %s: %d ferida(s), calibracao=%s",
                caminho.name, len(feridas), status_calibracao)

    return ResultadoAnalise(
        caminho_imagem=caminho,
        imagem_rgb=imagem_rgb,
        mascara=mascara,
        sobreposicao=sobreposicao,
        anotada=anotada,
        feridas=feridas,
        escala=escala,
        status_calibracao=status_calibracao,
        checkpoint=caminho_modelo,
    )


def linhas_de_medidas(resultado: ResultadoAnalise) -> List[Tuple[str, str, str]]:
    """Monta as linhas ``(medida, pixels, centimetros)`` da ferida principal.

    Quando não há escala, a coluna de centímetros traz um traço — e não um
    número estimado.
    """
    f = resultado.principal
    INDISPONIVEL = "—"

    def cm(valor: Optional[float], formato: str) -> str:
        return formato % valor if valor is not None else INDISPONIVEL

    return [
        ("Área", "%.0f px²" % f.area_pixels, cm(f.area_cm2, "%.3f cm²")),
        ("Comprimento", "%.1f px" % f.length_pixels, cm(f.length_cm, "%.2f cm")),
        ("Largura", "%.1f px" % f.width_pixels, cm(f.width_cm, "%.2f cm")),
        ("Perímetro", "%.1f px" % f.perimeter_pixels, cm(f.perimeter_cm, "%.2f cm")),
    ]
