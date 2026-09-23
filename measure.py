"""
measure.py
==========
Etapas 9, 10 e 11 — Medição morfológica da ferida a partir da máscara
segmentada, conversão opcional para unidades físicas reais, e geração
de relatório tabular (DataFrame / CSV).

Fluxo implementado:

    imagem -> segmentação -> máscara -> pós-processamento (limpeza)
           -> componentes conectados -> medição por ferida
           -> conversão opcional para cm -> relatório / visualização

Duas APIs convivem:

- :func:`measure_wound` (legado, mantida sem alterações de
  comportamento): mede o **maior** componente conectado da máscara
  crua, sem pós-processamento, e levanta ``ValueError`` se a máscara
  estiver vazia.
- :func:`measure_wounds` (nova): aplica o pós-processamento
  configurável de :class:`config.MeasurementConfig` e devolve **uma
  medição por ferida** encontrada, ordenadas da maior para a menor.
  Devolve lista vazia (sem exceção) quando não há ferida.

Todas as medidas são calculadas em pixels da **imagem original**. A
conversão para unidades físicas só acontece quando um fator de escala
explícito é fornecido (:class:`PixelScale`); o projeto nunca infere
escala sozinho.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

import config
from utils import draw_contour, get_logger, overlay_mask_on_image

logger = get_logger(__name__, config.LOGS_DIR / "measure.log")

CM_PER_INCH = 2.54

# Campos usados apenas para desenho — omitidos das linhas de relatório
# para manter os CSVs legíveis.
_DRAWING_FIELDS = ("rotated_box", "length_axis", "width_axis")


# --------------------------------------------------------------------------- #
# Etapa 10 — Escala pixel -> unidade física
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PixelScale:
    """Fator de conversão entre pixels da imagem original e centímetros.

    A unidade interna canônica é ``pixels_per_cm``; helpers para
    milímetros e polegadas são derivados dela, de modo que adicionar
    novas unidades no futuro não exige mudar as medições.

    Args:
        pixels_per_cm: quantos pixels correspondem a 1 cm na imagem.

    Raises:
        ValueError: se ``pixels_per_cm`` não for estritamente positivo.
    """

    pixels_per_cm: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.pixels_per_cm) or self.pixels_per_cm <= 0:
            raise ValueError("pixels_por_cm deve ser um valor positivo e finito.")

    # -- construtores alternativos ----------------------------------------- #

    @classmethod
    def from_pixels_per_cm(cls, pixels_per_cm: float) -> "PixelScale":
        """Cria a escala a partir de pixels por centímetro."""
        return cls(float(pixels_per_cm))

    @classmethod
    def from_pixels_per_mm(cls, pixels_per_mm: float) -> "PixelScale":
        """Cria a escala a partir de pixels por milímetro."""
        return cls(float(pixels_per_mm) * 10.0)

    @classmethod
    def from_reference_object(cls, reference_pixels: float, reference_cm: float) -> "PixelScale":
        """Deriva a escala de um objeto de referência presente na foto.

        Abordagem recomendada em campo: fotografar a ferida com um
        marcador de tamanho conhecido (régua descartável, adesivo
        circular, ou mesmo uma moeda) **no mesmo plano da ferida**.

        Args:
            reference_pixels: comprimento do objeto medido na imagem, em pixels.
            reference_cm: comprimento real desse objeto, em centímetros.

        Returns:
            Instância de :class:`PixelScale`.

        Raises:
            ValueError: se algum dos valores não for positivo.
        """
        if reference_pixels <= 0 or reference_cm <= 0:
            raise ValueError("Referência inválida: use valores positivos em pixels e cm.")
        return cls(float(reference_pixels) / float(reference_cm))

    @classmethod
    def from_dpi(cls, dpi: float) -> "PixelScale":
        """Deriva a escala de uma resolução em DPI (escâneres/planares)."""
        if dpi <= 0:
            raise ValueError("dpi deve ser positivo.")
        return cls(float(dpi) / CM_PER_INCH)

    @classmethod
    def resolve(cls, scale: Union["PixelScale", float, int, None]) -> Optional["PixelScale"]:
        """Normaliza o argumento de escala aceito pelas funções públicas.

        Aceita uma :class:`PixelScale`, um número (interpretado como
        pixels por cm, mantendo compatibilidade com o parâmetro
        ``pixels_por_cm`` já existente) ou ``None``.

        Args:
            scale: valor a normalizar.

        Returns:
            :class:`PixelScale` ou ``None`` quando não há escala.
        """
        if scale is None:
            return None
        if isinstance(scale, PixelScale):
            return scale
        return cls.from_pixels_per_cm(float(scale))

    # -- conversões --------------------------------------------------------- #

    @property
    def pixels_per_mm(self) -> float:
        """Pixels equivalentes a 1 mm."""
        return self.pixels_per_cm / 10.0

    def length_to_cm(self, pixels: float) -> float:
        """Converte um comprimento em pixels para centímetros."""
        return float(pixels) / self.pixels_per_cm

    def length_to_mm(self, pixels: float) -> float:
        """Converte um comprimento em pixels para milímetros."""
        return self.length_to_cm(pixels) * 10.0

    def area_to_cm2(self, pixels2: float) -> float:
        """Converte uma área em pixels² para cm²."""
        return float(pixels2) / (self.pixels_per_cm ** 2)

    def area_to_mm2(self, pixels2: float) -> float:
        """Converte uma área em pixels² para mm²."""
        return self.area_to_cm2(pixels2) * 100.0


def scale_from_options(
    pixels_por_cm: Optional[float] = None,
    pixels_por_mm: Optional[float] = None,
    referencia_pixels: Optional[float] = None,
    referencia_cm: Optional[float] = None,
) -> Optional[PixelScale]:
    """Monta uma :class:`PixelScale` a partir das opções de linha de comando.

    A precedência é: ``pixels_por_cm`` > ``pixels_por_mm`` > objeto de
    referência. Se nenhuma opção for informada, cai em
    ``config.DEFAULT_PIXELS_POR_CM`` e, na ausência dele, retorna
    ``None`` (medidas permanecem em pixels).

    Args:
        pixels_por_cm: fator direto pixels/cm.
        pixels_por_mm: fator direto pixels/mm.
        referencia_pixels: tamanho do objeto de referência na imagem (px).
        referencia_cm: tamanho real do objeto de referência (cm).

    Returns:
        :class:`PixelScale` ou ``None``.
    """
    if pixels_por_cm:
        return PixelScale.from_pixels_per_cm(pixels_por_cm)
    if pixels_por_mm:
        return PixelScale.from_pixels_per_mm(pixels_por_mm)
    if referencia_pixels and referencia_cm:
        return PixelScale.from_reference_object(referencia_pixels, referencia_cm)
    if config.DEFAULT_PIXELS_POR_CM:
        return PixelScale.from_pixels_per_cm(config.DEFAULT_PIXELS_POR_CM)
    return None


# --------------------------------------------------------------------------- #
# Estrutura de resultado
# --------------------------------------------------------------------------- #

@dataclass
class WoundMeasurements:
    """Conjunto completo de medidas morfológicas extraídas de uma máscara.

    Os campos até ``perimeter_cm`` são os originais do projeto e mantêm
    exatamente a mesma semântica. Os demais foram acrescentados pela
    etapa de medição e possuem valores padrão, de modo que qualquer
    código que já construía ou consumia esta classe continua válido.
    """

    area_pixels: int
    perimeter_pixels: float
    bounding_box: tuple  # (x, y, w, h) — alinhada aos eixos da imagem
    centroid: tuple  # (cx, cy)
    circularity: float
    aspect_ratio: float
    convex_area_pixels: float
    solidity: float
    percent_image_area: float
    num_connected_components: int
    area_cm2: Optional[float] = None
    perimeter_cm: Optional[float] = None

    # --- identificação (múltiplas feridas na mesma imagem) ----------------- #
    wound_id: int = 1

    # --- dimensões invariantes à rotação (retângulo de área mínima) -------- #
    length_pixels: float = 0.0  # maior lado do retângulo mínimo
    width_pixels: float = 0.0  # menor lado (largura perpendicular)
    feret_max_pixels: float = 0.0  # maior distância entre dois pontos da borda
    equivalent_diameter_pixels: float = 0.0  # diâmetro do círculo de mesma área
    length_width_ratio: float = 0.0
    extent: float = 0.0  # área / área da bounding box
    orientation_deg: float = 0.0  # ângulo do eixo de maior comprimento

    # --- geometria auxiliar para desenho ----------------------------------- #
    rotated_box: tuple = field(default_factory=tuple)  # 4 vértices (x, y)
    length_axis: tuple = field(default_factory=tuple)  # ((x1, y1), (x2, y2))
    width_axis: tuple = field(default_factory=tuple)  # ((x1, y1), (x2, y2))

    # --- unidades físicas (somente quando há escala confiável) ------------- #
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    feret_max_cm: Optional[float] = None
    equivalent_diameter_cm: Optional[float] = None
    bbox_width_cm: Optional[float] = None
    bbox_height_cm: Optional[float] = None
    pixels_por_cm: Optional[float] = None

    @property
    def has_physical_units(self) -> bool:
        """``True`` se as medidas em cm foram efetivamente calculadas."""
        return self.pixels_por_cm is not None


# --------------------------------------------------------------------------- #
# Etapa 9a — Pós-processamento da máscara
# --------------------------------------------------------------------------- #

def _binarize(mask: np.ndarray) -> np.ndarray:
    """Converte qualquer máscara (0/1, 0/255, float, bool) em uint8 0/1."""
    return (np.asarray(mask) > 0).astype(np.uint8)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Preenche buracos internos de uma máscara binária.

    Reconstrói cada região a partir de seu contorno externo, o que
    elimina falsos "furos" gerados pela segmentação (tecidos de aspecto
    diferente dentro da ferida classificados como fundo). Sem isso, a
    área e o perímetro ficariam subestimados/superestimados.

    Args:
        mask: máscara binária (0/1 ou 0/255).

    Returns:
        Máscara binária 0/1 sem buracos internos.
    """
    mask_bin = _binarize(mask)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bin
    filled = np.zeros_like(mask_bin)
    cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)
    return filled


def clean_mask(
    mask: np.ndarray,
    cfg: Optional[config.MeasurementConfig] = None,
) -> np.ndarray:
    """Aplica o pós-processamento que antecede a medição.

    Ordem das operações e justificativa:

    1. **Abertura morfológica** (desativada por padrão) — removeria
       pixels isolados, mas eroda a borda e enviesa área/perímetro;
       disponível via ``opening_kernel_size`` quando as máscaras vêm
       muito serrilhadas.
    2. **Fechamento morfológico** (desativado por padrão) — reconecta
       falhas finas, com a mesma ressalva de viés de borda.
    3. **Preenchimento de buracos** — ver :func:`fill_mask_holes`.
       Altera apenas o interior; a borda externa é preservada.
    4. **Remoção de componentes pequenos** — descarta regiões cuja área
       é irrelevante frente ao tamanho da imagem, que quase sempre são
       falsos positivos e distorceriam a contagem de feridas. É o
       mecanismo padrão de remoção de ruído justamente por ser exato e
       não deslocar a borda da ferida.

    Args:
        mask: máscara binária (H, W).
        cfg: configuração de medição; usa ``config.MEASURE_CFG`` se ``None``.

    Returns:
        Máscara binária 0/1 limpa, com o mesmo shape da entrada.
    """
    cfg = cfg or config.MEASURE_CFG
    mask_bin = _binarize(mask)
    if mask_bin.sum() == 0:
        return mask_bin

    if cfg.opening_kernel_size and cfg.opening_kernel_size >= 3:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.opening_kernel_size, cfg.opening_kernel_size)
        )
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k)

    if cfg.closing_kernel_size and cfg.closing_kernel_size >= 3:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.closing_kernel_size, cfg.closing_kernel_size)
        )
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k)

    if cfg.fill_holes:
        mask_bin = fill_mask_holes(mask_bin)

    return _remove_small_components(mask_bin, cfg)


def _min_component_area(mask_shape: Tuple[int, int], cfg: config.MeasurementConfig) -> float:
    """Área mínima (px) para um componente ser aceito como ferida."""
    total = float(mask_shape[0] * mask_shape[1])
    return max(float(cfg.min_component_area_px), cfg.min_component_area_ratio * total)


def _remove_small_components(mask: np.ndarray, cfg: config.MeasurementConfig) -> np.ndarray:
    """Zera os componentes conectados menores que o limiar de área."""
    mask_bin = _binarize(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=cfg.connectivity
    )
    if num_labels <= 1:
        return mask_bin

    min_area = _min_component_area(mask_bin.shape, cfg)
    keep = np.zeros(num_labels, dtype=bool)
    for label in range(1, num_labels):
        keep[label] = stats[label, cv2.CC_STAT_AREA] >= min_area

    if not keep.any():
        # Nenhum componente atinge o limiar: preserva o maior para não
        # descartar uma ferida legítima em imagem de baixa resolução.
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep[largest] = True

    return keep[labels].astype(np.uint8)


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Retorna uma máscara binária contendo apenas o maior componente conectado.

    Args:
        mask: máscara binária (0/1 ou 0/255).

    Returns:
        Máscara binária (0/1) com apenas o maior componente conectado.
    """
    mask_bin = _binarize(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:  # apenas o fundo
        return mask_bin

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Etapa 9b — Extração de características geométricas
# --------------------------------------------------------------------------- #

def _feret_max(contour: np.ndarray) -> float:
    """Maior distância entre dois pontos da borda (diâmetro de Feret máximo).

    Calculada sobre o fecho convexo, que contém necessariamente o par de
    pontos mais distantes — reduz o custo de O(n²) sobre o contorno
    completo para O(h²) com h = número de vértices do hull.
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        return 0.0
    diffs = hull[:, None, :] - hull[None, :, :]
    return float(np.sqrt((diffs ** 2).sum(axis=2)).max())


def _rotated_dimensions(contour: np.ndarray) -> dict:
    """Extrai comprimento, largura, orientação e eixos do retângulo mínimo.

    O retângulo de área mínima (``cv2.minAreaRect``) é invariante à
    rotação da ferida na foto: o maior lado corresponde ao maior
    comprimento e o menor lado à largura perpendicular a ele — a mesma
    convenção usada na aferição clínica manual.

    Returns:
        Dicionário com ``length``, ``width``, ``orientation_deg``,
        ``box`` (4 vértices), ``length_axis`` e ``width_axis``.
    """
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    length, width = (rw, rh) if rw >= rh else (rh, rw)

    # Direção do maior lado
    theta = math.radians(angle if rw >= rh else angle + 90.0)
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -uy, ux  # perpendicular

    length_axis = (
        (cx - ux * length / 2.0, cy - uy * length / 2.0),
        (cx + ux * length / 2.0, cy + uy * length / 2.0),
    )
    width_axis = (
        (cx - vx * width / 2.0, cy - vy * width / 2.0),
        (cx + vx * width / 2.0, cy + vy * width / 2.0),
    )

    box = tuple(tuple(float(v) for v in pt) for pt in cv2.boxPoints(((cx, cy), (rw, rh), angle)))

    return {
        "length": float(length),
        "width": float(width),
        "orientation_deg": float(math.degrees(theta) % 180.0),
        "box": box,
        "length_axis": length_axis,
        "width_axis": width_axis,
    }


def _measure_component(
    component_mask: np.ndarray,
    total_pixels: int,
    num_connected_components: int,
    wound_id: int = 1,
) -> Optional[WoundMeasurements]:
    """Mede um único componente conectado já isolado.

    Args:
        component_mask: máscara binária 0/1 contendo **uma** região.
        total_pixels: número de pixels da imagem inteira (para o percentual).
        num_connected_components: total de feridas na imagem (informativo).
        wound_id: identificador sequencial da ferida.

    Returns:
        :class:`WoundMeasurements` ou ``None`` se o componente não gerar
        contorno válido.
    """
    area = int(component_mask.sum())
    if area == 0:
        return None

    contours, _ = cv2.findContours(
        component_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)

    perimeter = float(cv2.arcLength(largest_contour, closed=True))
    x, y, w, h = cv2.boundingRect(largest_contour)

    moments = cv2.moments(largest_contour)
    cx = moments["m10"] / moments["m00"] if moments["m00"] != 0 else x + w / 2
    cy = moments["m01"] / moments["m00"] if moments["m00"] != 0 else y + h / 2

    circularity = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    aspect_ratio = float(w / h) if h > 0 else 0.0

    hull = cv2.convexHull(largest_contour)
    convex_area = float(cv2.contourArea(hull))
    solidity = float(area / convex_area) if convex_area > 0 else 0.0

    rot = _rotated_dimensions(largest_contour)
    equivalent_diameter = float(math.sqrt(4.0 * area / math.pi))
    bbox_area = float(w * h)

    return WoundMeasurements(
        area_pixels=area,
        perimeter_pixels=perimeter,
        bounding_box=(int(x), int(y), int(w), int(h)),
        centroid=(float(cx), float(cy)),
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        convex_area_pixels=convex_area,
        solidity=solidity,
        percent_image_area=100.0 * area / total_pixels,
        num_connected_components=num_connected_components,
        wound_id=int(wound_id),
        length_pixels=rot["length"],
        width_pixels=rot["width"],
        feret_max_pixels=_feret_max(largest_contour),
        equivalent_diameter_pixels=equivalent_diameter,
        length_width_ratio=float(rot["length"] / rot["width"]) if rot["width"] > 0 else 0.0,
        extent=float(area / bbox_area) if bbox_area > 0 else 0.0,
        orientation_deg=rot["orientation_deg"],
        rotated_box=rot["box"],
        length_axis=rot["length_axis"],
        width_axis=rot["width_axis"],
    )


# --------------------------------------------------------------------------- #
# Etapa 9c — APIs públicas de medição
# --------------------------------------------------------------------------- #

def measure_wound(
    mask: np.ndarray,
    pixels_por_cm: Union[PixelScale, float, None] = None,
) -> WoundMeasurements:
    """Calcula as medidas morfológicas da ferida a partir de uma máscara binária.

    Mede o **maior** componente conectado da máscara, sem pós-processamento
    (comportamento histórico preservado). Para medir todas as feridas de
    uma imagem, use :func:`measure_wounds`.

    Args:
        mask: máscara binária (H, W), com valores 0/1 ou 0/255.
        pixels_por_cm: fator de escala opcional (número em pixels/cm ou
            uma instância de :class:`PixelScale`). Se informado, as
            medidas também são retornadas em cm/cm².

    Returns:
        Instância de :class:`WoundMeasurements` com todas as métricas.

    Raises:
        ValueError: se a máscara não contiver nenhum pixel de primeiro plano.
    """
    mask_bin = _binarize(mask)
    total_pixels = mask_bin.shape[0] * mask_bin.shape[1]

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    num_components = max(num_labels - 1, 0)

    if mask_bin.sum() == 0:
        raise ValueError("Máscara vazia: nenhuma região de ferida encontrada.")

    main_mask = _largest_component_mask(mask_bin)
    result = _measure_component(main_mask, total_pixels, num_components, wound_id=1)
    if result is None:  # pragma: no cover - máscara não vazia sempre gera contorno
        raise ValueError("Máscara vazia: nenhuma região de ferida encontrada.")

    scale = PixelScale.resolve(pixels_por_cm)
    if scale is not None:
        result = _apply_scale(result, scale)

    return result


def measure_wounds(
    mask: np.ndarray,
    pixels_por_cm: Union[PixelScale, float, None] = None,
    cfg: Optional[config.MeasurementConfig] = None,
    cleanup: bool = True,
) -> List[WoundMeasurements]:
    """Mede **todas** as feridas presentes na máscara, uma a uma.

    Cada componente conectado sobrevivente ao pós-processamento é tratado
    como uma ferida independente e recebe um ``wound_id`` sequencial,
    atribuído em ordem decrescente de área.

    Args:
        mask: máscara binária (H, W), com valores 0/1 ou 0/255.
        pixels_por_cm: fator de escala opcional (pixels/cm ou :class:`PixelScale`).
        cfg: configuração de pós-processamento; usa ``config.MEASURE_CFG`` se ``None``.
        cleanup: se ``False``, mede a máscara crua sem limpeza alguma.

    Returns:
        Lista de :class:`WoundMeasurements` ordenada da maior para a
        menor área. **Lista vazia** quando nenhuma ferida é detectada —
        esta função não levanta exceção nesse caso.
    """
    cfg = cfg or config.MEASURE_CFG
    mask_bin = _binarize(mask)
    total_pixels = mask_bin.shape[0] * mask_bin.shape[1]
    if total_pixels == 0 or mask_bin.sum() == 0:
        return []

    processed = clean_mask(mask_bin, cfg) if cleanup else mask_bin
    if processed.sum() == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        processed, connectivity=cfg.connectivity
    )
    if num_labels <= 1:
        return []

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]  # maior primeiro
    if cfg.max_wounds is not None:
        order = order[: cfg.max_wounds]

    num_components = int(num_labels - 1)
    scale = PixelScale.resolve(pixels_por_cm)

    results: List[WoundMeasurements] = []
    for wound_id, idx in enumerate(order, start=1):
        label = int(idx) + 1
        component = (labels == label).astype(np.uint8)
        measurement = _measure_component(component, total_pixels, num_components, wound_id)
        if measurement is None:
            continue
        if scale is not None:
            measurement = _apply_scale(measurement, scale)
        results.append(measurement)

    if cfg.max_wounds is not None and num_components > len(results):
        logger.info(
            "%d componentes detectados; reportando as %d maiores feridas "
            "(limite config.MEASURE_CFG.max_wounds).",
            num_components, len(results),
        )
    return results


def _apply_scale(
    measurements: WoundMeasurements,
    pixels_por_cm: Union[PixelScale, float],
) -> WoundMeasurements:
    """Etapa 10 — Converte as medidas de pixels para cm²/cm.

    Args:
        measurements: medidas originais em pixels.
        pixels_por_cm: número de pixels equivalentes a 1 cm, ou uma
            instância de :class:`PixelScale`.

    Returns:
        A mesma instância de :class:`WoundMeasurements`, com os campos
        em unidades físicas preenchidos.

    Raises:
        ValueError: se o fator de escala não for positivo.
    """
    scale = PixelScale.resolve(pixels_por_cm)
    if scale is None:  # pragma: no cover - resolve() só devolve None para None
        raise ValueError("pixels_por_cm deve ser um valor positivo.")

    _, _, bw, bh = measurements.bounding_box

    measurements.area_cm2 = scale.area_to_cm2(measurements.area_pixels)
    measurements.perimeter_cm = scale.length_to_cm(measurements.perimeter_pixels)
    measurements.length_cm = scale.length_to_cm(measurements.length_pixels)
    measurements.width_cm = scale.length_to_cm(measurements.width_pixels)
    measurements.feret_max_cm = scale.length_to_cm(measurements.feret_max_pixels)
    measurements.equivalent_diameter_cm = scale.length_to_cm(
        measurements.equivalent_diameter_pixels
    )
    measurements.bbox_width_cm = scale.length_to_cm(bw)
    measurements.bbox_height_cm = scale.length_to_cm(bh)
    measurements.pixels_por_cm = scale.pixels_per_cm
    return measurements


# --------------------------------------------------------------------------- #
# Etapa 11 — Relatório tabular
# --------------------------------------------------------------------------- #

def measurements_to_row(image_name: str, measurements: WoundMeasurements, **extra) -> dict:
    """Converte um :class:`WoundMeasurements` em uma linha de DataFrame (dict).

    Os campos usados apenas para desenho (``rotated_box``, ``length_axis``,
    ``width_axis``) são omitidos para manter o CSV legível.

    Args:
        image_name: nome do arquivo de imagem correspondente.
        measurements: medidas calculadas por :func:`measure_wound`.
        **extra: campos adicionais (ex.: ``iou``, ``dice``) a incluir na linha.

    Returns:
        Dicionário representando uma linha do relatório final.
    """
    row = {"nome_imagem": image_name}
    row.update({k: v for k, v in asdict(measurements).items() if k not in _DRAWING_FIELDS})
    row.update(extra)
    return row


def wounds_to_rows(
    image_name: str, wounds: Sequence[WoundMeasurements], **extra
) -> List[dict]:
    """Converte uma lista de medições (uma por ferida) em linhas de relatório.

    Args:
        image_name: nome do arquivo de imagem correspondente.
        wounds: medições devolvidas por :func:`measure_wounds`.
        **extra: campos adicionais replicados em todas as linhas.

    Returns:
        Lista de dicionários — uma linha por ferida. Quando ``wounds``
        está vazia, devolve uma única linha sinalizando a ausência de
        ferida, para que a imagem não desapareça do relatório.
    """
    if not wounds:
        return [{
            "nome_imagem": image_name,
            "wound_id": 0,
            "observacao": "nenhuma ferida detectada",
            **extra,
        }]
    return [measurements_to_row(image_name, w, **extra) for w in wounds]


def format_measurements_text(
    wounds: Sequence[WoundMeasurements], decimals: int = 1
) -> str:
    """Formata as medições para exibição em texto (console/log/relatório).

    Mostra as medidas em cm apenas quando existe escala confiável; caso
    contrário, apresenta somente pixels e informa explicitamente que não
    há conversão física disponível.

    Args:
        wounds: medições devolvidas por :func:`measure_wounds`.
        decimals: casas decimais usadas nos valores.

    Returns:
        Bloco de texto pronto para impressão.
    """
    if not wounds:
        return "Nenhuma ferida detectada na imagem."

    lines: List[str] = []
    for w in wounds:
        lines.append(f"Ferida {w.wound_id}")
        if w.has_physical_units:
            lines.append(f"  Área: {w.area_cm2:.{decimals}f} cm² ({w.area_pixels} px²)")
            lines.append(
                f"  Comprimento: {w.length_cm:.{decimals}f} cm "
                f"({w.length_pixels:.{decimals}f} px)"
            )
            lines.append(
                f"  Largura: {w.width_cm:.{decimals}f} cm "
                f"({w.width_pixels:.{decimals}f} px)"
            )
            lines.append(
                f"  Perímetro: {w.perimeter_cm:.{decimals}f} cm "
                f"({w.perimeter_pixels:.{decimals}f} px)"
            )
        else:
            lines.append(f"  Área: {w.area_pixels} px²")
            lines.append(f"  Comprimento: {w.length_pixels:.{decimals}f} px")
            lines.append(f"  Largura: {w.width_pixels:.{decimals}f} px")
            lines.append(f"  Perímetro: {w.perimeter_pixels:.{decimals}f} px")
        lines.append(
            f"  Bounding box (x, y, w, h): {w.bounding_box} | "
            f"Centroide: ({w.centroid[0]:.{decimals}f}, {w.centroid[1]:.{decimals}f})"
        )
        lines.append(
            f"  Circularidade: {w.circularity:.3f} | Solidez: {w.solidity:.3f} | "
            f"Razão C/L: {w.length_width_ratio:.3f}"
        )
        lines.append("")

    if not wounds[0].has_physical_units:
        lines.append(
            "Sem fator de escala informado: todas as medidas estão em pixels "
            "(use --pixels-por-cm / --referencia-pixels + --referencia-cm)."
        )
    return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------- #
# Etapa 11b — Visualização das medições
# --------------------------------------------------------------------------- #

_PALETTE = (
    (255, 215, 0),    # amarelo — bounding box
    (0, 255, 255),    # ciano — retângulo mínimo
    (255, 64, 64),    # vermelho — comprimento
    (64, 128, 255),   # azul — largura
    (0, 255, 0),      # verde — contorno
)


def annotate_measurements(
    image: np.ndarray,
    wounds: Sequence[WoundMeasurements],
    mask: Optional[np.ndarray] = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Desenha contorno, caixas, eixos e valores das medidas sobre a imagem.

    Elementos desenhados por ferida:
        - contorno (verde) e overlay translúcido da máscara, se fornecida;
        - bounding box alinhada aos eixos (amarelo);
        - retângulo de área mínima (ciano);
        - linha do comprimento (vermelho) e da largura (azul);
        - rótulo com id, área, comprimento e largura.

    Args:
        image: imagem RGB uint8 (H, W, 3).
        wounds: medições devolvidas por :func:`measure_wounds`.
        mask: máscara binária usada para o overlay/contorno (opcional).
        draw_labels: se ``False``, desenha apenas a geometria.

    Returns:
        Nova imagem RGB uint8 com as anotações.
    """
    canvas = image.copy()
    if mask is not None:
        canvas = overlay_mask_on_image(canvas, mask)
        canvas = draw_contour(canvas, mask, color=_PALETTE[4], thickness=2)

    scale_factor = max(1.0, min(canvas.shape[:2]) / 512.0)
    thickness = max(1, int(round(2 * scale_factor)))
    font_scale = 0.55 * scale_factor

    for w in wounds:
        x, y, bw, bh = w.bounding_box
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), _PALETTE[0], thickness)

        if w.rotated_box:
            box = np.array(w.rotated_box, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [box], isClosed=True, color=_PALETTE[1], thickness=thickness)

        for axis, color in ((w.length_axis, _PALETTE[2]), (w.width_axis, _PALETTE[3])):
            if not axis:
                continue
            p1 = (int(round(axis[0][0])), int(round(axis[0][1])))
            p2 = (int(round(axis[1][0])), int(round(axis[1][1])))
            cv2.line(canvas, p1, p2, color, thickness)

        cx, cy = w.centroid
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), thickness + 2, _PALETTE[2], -1)

        if draw_labels:
            _draw_label_block(canvas, w, font_scale, thickness)

    return canvas


def _label_lines(w: WoundMeasurements) -> List[str]:
    """Linhas de texto exibidas ao lado de cada ferida anotada."""
    if w.has_physical_units:
        return [
            f"Ferida {w.wound_id}",
            f"A: {w.area_cm2:.2f} cm2",
            f"C: {w.length_cm:.2f} cm",
            f"L: {w.width_cm:.2f} cm",
            f"P: {w.perimeter_cm:.2f} cm",
        ]
    return [
        f"Ferida {w.wound_id}",
        f"A: {w.area_pixels} px2",
        f"C: {w.length_pixels:.0f} px",
        f"L: {w.width_pixels:.0f} px",
        f"P: {w.perimeter_pixels:.0f} px",
    ]


def _draw_label_block(
    canvas: np.ndarray, w: WoundMeasurements, font_scale: float, thickness: int
) -> None:
    """Desenha a caixa de texto com as medidas, mantendo-a dentro da imagem."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = _label_lines(w)
    text_thickness = max(1, thickness - 1)

    sizes = [cv2.getTextSize(line, font, font_scale, text_thickness)[0] for line in lines]
    box_w = max(s[0] for s in sizes) + 10
    line_h = max(s[1] for s in sizes) + 6
    box_h = line_h * len(lines) + 6

    x, y, _, bh = w.bounding_box
    h_img, w_img = canvas.shape[:2]
    ox = int(np.clip(x, 0, max(w_img - box_w, 0)))
    oy = y - box_h - 4
    if oy < 0:  # não cabe acima da caixa: desenha abaixo
        oy = min(y + bh + 4, max(h_img - box_h, 0))
    oy = int(np.clip(oy, 0, max(h_img - box_h, 0)))

    backdrop = canvas[oy:oy + box_h, ox:ox + box_w]
    if backdrop.size:
        canvas[oy:oy + box_h, ox:ox + box_w] = (0.45 * backdrop).astype(np.uint8)

    for i, line in enumerate(lines):
        baseline = oy + line_h * (i + 1)
        cv2.putText(
            canvas, line, (ox + 5, baseline), font, font_scale,
            (255, 255, 255), text_thickness, cv2.LINE_AA,
        )


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from dataset import list_image_mask_pairs

    pairs = list_image_mask_pairs()
    if pairs:
        _, mask_path = pairs[0]
        sample_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        measurements = measure_wounds(sample_mask, pixels_por_cm=config.DEFAULT_PIXELS_POR_CM)
        logger.info("Medição de exemplo (%s):\n%s", mask_path.name,
                    format_measurements_text(measurements))
