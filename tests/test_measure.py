"""
test_measure.py
===============
Testes da etapa de medição (:mod:`measure`).

Cobrem os cenários exigidos pela especificação:
    1. imagem com uma única ferida;
    2. imagem com múltiplas feridas;
    3. máscara de formato irregular;
    4. máscara com pequenas regiões de ruído;
    5. imagem sem escala física disponível.

Além disso validam a compatibilidade com a API antiga
(:func:`measure.measure_wound`), a conversão de unidades e o
comportamento quando nenhuma ferida é detectada.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import config
from measure import (
    PixelScale,
    WoundMeasurements,
    annotate_measurements,
    clean_mask,
    fill_mask_holes,
    format_measurements_text,
    measure_wound,
    measure_wounds,
    measurements_to_row,
    scale_from_options,
    wounds_to_rows,
)


# --------------------------------------------------------------------------- #
# 1. Uma única ferida
# --------------------------------------------------------------------------- #

def test_single_wound_rect_dimensions(single_rect_mask):
    """Retângulo 120x60: área, comprimento, largura e bbox exatos."""
    wounds = measure_wounds(single_rect_mask)

    assert len(wounds) == 1
    w = wounds[0]
    assert w.wound_id == 1
    assert w.area_pixels == 120 * 60
    assert w.bounding_box == (100, 80, 120, 60)
    assert w.length_pixels == pytest.approx(119.0, abs=1.5)
    assert w.width_pixels == pytest.approx(59.0, abs=1.5)
    assert w.length_pixels >= w.width_pixels
    assert w.length_width_ratio == pytest.approx(2.0, rel=0.05)


def test_single_wound_perimeter_and_extent(single_rect_mask):
    """Perímetro do retângulo ~2*(119+59); extent ~1 (preenche a bbox)."""
    w = measure_wounds(single_rect_mask)[0]

    assert w.perimeter_pixels == pytest.approx(2 * (119 + 59), rel=0.05)
    assert w.extent == pytest.approx(1.0, abs=0.02)
    assert w.solidity == pytest.approx(1.0, abs=0.05)


def test_single_wound_centroid(single_rect_mask):
    """Centroide no centro geométrico do retângulo."""
    w = measure_wounds(single_rect_mask)[0]

    assert w.centroid[0] == pytest.approx(159.5, abs=1.0)  # x
    assert w.centroid[1] == pytest.approx(109.5, abs=1.0)  # y


def test_circle_metrics(single_circle_mask):
    """Círculo r=70: área ~pi*r², circularidade ~1, C/L ~1."""
    w = measure_wounds(single_circle_mask)[0]

    assert w.area_pixels == pytest.approx(math.pi * 70 ** 2, rel=0.02)
    assert w.equivalent_diameter_pixels == pytest.approx(140.0, rel=0.02)
    assert w.length_width_ratio == pytest.approx(1.0, abs=0.05)
    # arcLength sobre a borda em escada superestima levemente o perímetro,
    # então a circularidade fica um pouco abaixo de 1.
    assert 0.85 <= w.circularity <= 1.05


def test_feret_max_is_at_least_length(single_rect_mask, irregular_mask):
    """O diâmetro de Feret máximo nunca é menor que o comprimento medido."""
    for mask in (single_rect_mask, irregular_mask):
        w = measure_wounds(mask)[0]
        assert w.feret_max_pixels >= w.length_pixels - 1e-6


# --------------------------------------------------------------------------- #
# 2. Múltiplas feridas
# --------------------------------------------------------------------------- #

def test_multiple_wounds_are_measured_individually(two_wounds_mask):
    """Duas feridas separadas geram duas medições ordenadas por área."""
    wounds = measure_wounds(two_wounds_mask)

    assert len(wounds) == 2
    assert [w.wound_id for w in wounds] == [1, 2]
    assert wounds[0].area_pixels > wounds[1].area_pixels
    assert wounds[0].area_pixels == 80 * 100
    assert wounds[1].area_pixels == 40 * 60
    assert all(w.num_connected_components == 2 for w in wounds)


def test_multiple_wounds_have_distinct_centroids(two_wounds_mask):
    """Cada ferida tem geometria própria — nada é agregado."""
    w1, w2 = measure_wounds(two_wounds_mask)

    assert w1.bounding_box == (30, 30, 100, 80)
    assert w2.bounding_box == (250, 250, 60, 40)
    assert w1.centroid != w2.centroid


def test_legacy_measure_wound_picks_largest(two_wounds_mask):
    """A API antiga continua devolvendo o maior componente."""
    legacy = measure_wound(two_wounds_mask)
    biggest = measure_wounds(two_wounds_mask)[0]

    assert legacy.area_pixels == biggest.area_pixels
    assert legacy.bounding_box == biggest.bounding_box
    assert legacy.num_connected_components == 2


def test_max_wounds_limit_is_respected(monkeypatch):
    """``max_wounds`` limita quantas feridas são reportadas."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    for i in range(5):
        y = 20 + i * 70
        mask[y:y + 40, 20:80] = 1

    cfg = config.MeasurementConfig(max_wounds=3)
    wounds = measure_wounds(mask, cfg=cfg)

    assert len(wounds) == 3
    assert all(w.num_connected_components == 5 for w in wounds)


# --------------------------------------------------------------------------- #
# 3. Formato irregular
# --------------------------------------------------------------------------- #

def test_irregular_shape_metrics(irregular_mask):
    """Ferida em "L": solidez < 1 e circularidade baixa, sem quebrar."""
    wounds = measure_wounds(irregular_mask)

    assert len(wounds) == 1
    w = wounds[0]
    expected_area = 160 * 60 + 60 * 180 - 60 * 60  # braços menos a interseção
    assert w.area_pixels == pytest.approx(expected_area, rel=0.02)
    assert w.solidity < 0.85
    assert w.circularity < 0.6
    assert w.extent < 0.8
    assert w.length_pixels > 0 and w.width_pixels > 0


def test_rotation_invariance(rotated_rect_mask):
    """Retângulo 140x50 rotacionado 30°: comprimento/largura preservados."""
    w = measure_wounds(rotated_rect_mask)[0]

    assert w.length_pixels == pytest.approx(140.0, rel=0.05)
    assert w.width_pixels == pytest.approx(50.0, rel=0.08)
    # A bounding box alinhada aos eixos é maior que o retângulo real —
    # é justamente por isso que comprimento/largura vêm do minAreaRect.
    _, _, bw, bh = w.bounding_box
    assert bw > w.width_pixels and bh > w.width_pixels


def test_orientation_is_reported(rotated_rect_mask):
    """A orientação do eixo maior é reportada em [0, 180)."""
    w = measure_wounds(rotated_rect_mask)[0]
    assert 0.0 <= w.orientation_deg < 180.0


# --------------------------------------------------------------------------- #
# 4. Ruído, buracos e componentes desconectados
# --------------------------------------------------------------------------- #

def test_noise_is_removed_before_measuring(noisy_mask):
    """Ruído pontual não vira ferida; só a ferida real é reportada."""
    wounds = measure_wounds(noisy_mask)

    assert len(wounds) == 1
    assert wounds[0].bounding_box == (100, 80, 120, 60)


def test_holes_are_filled(noisy_mask):
    """O buraco interno de 6x6 é preenchido, restaurando a área real."""
    wounds = measure_wounds(noisy_mask)
    assert wounds[0].area_pixels == 120 * 60


def test_fill_mask_holes_directly():
    """`fill_mask_holes` remove buracos internos e preserva o exterior."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    mask[40:50, 40:50] = 0

    filled = fill_mask_holes(mask)

    assert filled.sum() == 60 * 60
    assert filled[0, 0] == 0


def test_cleanup_disabled_keeps_noise(noisy_mask):
    """Com ``cleanup=False`` o ruído permanece — o comportamento é opt-out."""
    wounds = measure_wounds(noisy_mask, cleanup=False)
    assert len(wounds) > 1


def test_legacy_measure_wound_ignores_cleanup(noisy_mask):
    """A API antiga não aplica limpeza: buraco e ruído continuam visíveis."""
    legacy = measure_wound(noisy_mask)

    assert legacy.area_pixels == 120 * 60 - 6 * 6  # buraco NÃO preenchido
    assert legacy.num_connected_components > 1     # ruído NÃO removido


def test_tiny_mask_is_not_discarded_entirely():
    """Se todos os componentes forem minúsculos, o maior é preservado."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[10:14, 10:14] = 1  # 16 px, abaixo do limiar de 50 px

    wounds = measure_wounds(mask)

    assert len(wounds) == 1
    assert wounds[0].area_pixels == 16


def test_cleaned_mask_measures_identically(noisy_mask):
    """Medir a máscara já limpa (``cleanup=False``) dá o mesmo resultado.

    É esse o contrato usado por ``predict``/``batch_inference`` para
    desenhar exatamente a máscara que foi medida, sem limpar duas vezes.
    """
    direct = measure_wounds(noisy_mask)
    two_step = measure_wounds(clean_mask(noisy_mask), cleanup=False)

    assert len(direct) == len(two_step)
    for a, b in zip(direct, two_step):
        assert a.area_pixels == b.area_pixels
        assert a.bounding_box == b.bounding_box
        assert a.perimeter_pixels == pytest.approx(b.perimeter_pixels)


def test_clean_mask_is_idempotent(noisy_mask):
    """Aplicar a limpeza duas vezes não altera mais a máscara."""
    once = clean_mask(noisy_mask)
    twice = clean_mask(once)
    assert np.array_equal(once, twice)


def test_clean_mask_preserves_shape(noisy_mask):
    """A limpeza não altera as dimensões da máscara."""
    cleaned = clean_mask(noisy_mask)
    assert cleaned.shape == noisy_mask.shape
    assert cleaned.dtype == np.uint8
    assert set(np.unique(cleaned)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# 5. Escala: pixels vs. unidades físicas
# --------------------------------------------------------------------------- #

def test_without_scale_physical_fields_are_none(single_rect_mask):
    """Sem escala, nenhum valor físico é inventado."""
    w = measure_wounds(single_rect_mask)[0]

    assert w.has_physical_units is False
    for field_name in (
        "area_cm2", "perimeter_cm", "length_cm", "width_cm",
        "feret_max_cm", "equivalent_diameter_cm",
        "bbox_width_cm", "bbox_height_cm", "pixels_por_cm",
    ):
        assert getattr(w, field_name) is None, field_name


def test_with_scale_conversion_is_correct(single_rect_mask):
    """Com 10 px/cm: 7200 px² -> 72 cm²; 120 px -> 12 cm."""
    w = measure_wounds(single_rect_mask, pixels_por_cm=10.0)[0]

    assert w.has_physical_units is True
    assert w.pixels_por_cm == 10.0
    assert w.area_cm2 == pytest.approx(72.0)
    assert w.length_cm == pytest.approx(w.length_pixels / 10.0)
    assert w.width_cm == pytest.approx(w.width_pixels / 10.0)
    assert w.perimeter_cm == pytest.approx(w.perimeter_pixels / 10.0)
    assert w.bbox_width_cm == pytest.approx(12.0)
    assert w.bbox_height_cm == pytest.approx(6.0)


def test_scale_accepts_pixelscale_instance(single_rect_mask):
    """`pixels_por_cm` também aceita um objeto :class:`PixelScale`."""
    scale = PixelScale.from_pixels_per_mm(1.0)  # 1 px/mm == 10 px/cm
    w = measure_wounds(single_rect_mask, pixels_por_cm=scale)[0]

    assert w.pixels_por_cm == pytest.approx(10.0)
    assert w.area_cm2 == pytest.approx(72.0)


def test_pixel_scale_factories():
    """Todos os construtores levam ao mesmo fator canônico."""
    assert PixelScale.from_pixels_per_cm(38.0).pixels_per_cm == 38.0
    assert PixelScale.from_pixels_per_mm(3.8).pixels_per_cm == pytest.approx(38.0)
    assert PixelScale.from_reference_object(190.0, 5.0).pixels_per_cm == pytest.approx(38.0)
    assert PixelScale.from_dpi(254.0).pixels_per_cm == pytest.approx(100.0)
    assert PixelScale(10.0).pixels_per_mm == pytest.approx(1.0)
    assert PixelScale(10.0).area_to_mm2(100.0) == pytest.approx(100.0)


def test_pixel_scale_rejects_invalid_values():
    """Escalas inválidas falham alto, em vez de gerar números sem sentido."""
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PixelScale(bad)
    with pytest.raises(ValueError):
        PixelScale.from_reference_object(100.0, 0.0)
    with pytest.raises(ValueError):
        PixelScale.from_dpi(-72.0)


def test_resolve_and_scale_from_options():
    """`resolve` e `scale_from_options` normalizam as várias entradas."""
    assert PixelScale.resolve(None) is None
    assert PixelScale.resolve(12.0).pixels_per_cm == 12.0
    scale = PixelScale(5.0)
    assert PixelScale.resolve(scale) is scale

    assert scale_from_options() is None
    assert scale_from_options(pixels_por_cm=20.0).pixels_per_cm == 20.0
    assert scale_from_options(pixels_por_mm=2.0).pixels_per_cm == pytest.approx(20.0)
    assert scale_from_options(
        referencia_pixels=100.0, referencia_cm=5.0
    ).pixels_per_cm == pytest.approx(20.0)


def test_default_scale_is_none():
    """O projeto não assume nenhuma escala física por padrão."""
    assert config.DEFAULT_PIXELS_POR_CM is None


# --------------------------------------------------------------------------- #
# 6. Ausência de ferida / robustez
# --------------------------------------------------------------------------- #

def test_empty_mask_returns_empty_list(empty_mask):
    """Sem ferida, `measure_wounds` devolve lista vazia sem levantar erro."""
    assert measure_wounds(empty_mask) == []
    assert measure_wounds(empty_mask, pixels_por_cm=10.0) == []


def test_empty_mask_legacy_raises(empty_mask):
    """A API antiga mantém o contrato de levantar ``ValueError``."""
    with pytest.raises(ValueError):
        measure_wound(empty_mask)


def test_clean_mask_on_empty(empty_mask):
    """A limpeza de uma máscara vazia devolve uma máscara vazia."""
    assert clean_mask(empty_mask).sum() == 0


def test_wounds_to_rows_when_no_wound(empty_mask):
    """A imagem sem ferida ainda aparece no relatório, sinalizada."""
    rows = wounds_to_rows("sem_ferida.png", measure_wounds(empty_mask))

    assert len(rows) == 1
    assert rows[0]["wound_id"] == 0
    assert rows[0]["observacao"] == "nenhuma ferida detectada"


def test_mask_with_255_values_is_accepted(single_rect_mask):
    """Máscaras 0/255 produzem o mesmo resultado que máscaras 0/1."""
    w_01 = measure_wounds(single_rect_mask)[0]
    w_255 = measure_wounds(single_rect_mask * 255)[0]

    assert w_01.area_pixels == w_255.area_pixels
    assert w_01.bounding_box == w_255.bounding_box


def test_boolean_and_float_masks_are_accepted(single_rect_mask):
    """Máscaras booleanas e float também são aceitas."""
    assert measure_wounds(single_rect_mask.astype(bool))[0].area_pixels == 7200
    assert measure_wounds(single_rect_mask.astype(np.float32))[0].area_pixels == 7200


def test_low_resolution_mask(single_rect_mask):
    """Máscara pequena (60x60) continua sendo medida sem quebrar."""
    import cv2

    small = cv2.resize(single_rect_mask, (60, 60), interpolation=cv2.INTER_NEAREST)
    wounds = measure_wounds(small)

    assert len(wounds) == 1
    assert wounds[0].area_pixels > 0
    assert wounds[0].length_pixels >= wounds[0].width_pixels


# --------------------------------------------------------------------------- #
# 7. Relatório e apresentação
# --------------------------------------------------------------------------- #

def test_measurements_to_row_excludes_drawing_fields(single_rect_mask):
    """Campos de desenho não poluem o CSV, mas as métricas estão lá."""
    row = measurements_to_row("img.png", measure_wounds(single_rect_mask)[0])

    assert row["nome_imagem"] == "img.png"
    for drawing_field in ("rotated_box", "length_axis", "width_axis"):
        assert drawing_field not in row
    for metric in ("area_pixels", "perimeter_pixels", "length_pixels",
                   "width_pixels", "circularity", "solidity", "wound_id"):
        assert metric in row


def test_wounds_to_rows_one_row_per_wound(two_wounds_mask):
    """Cada ferida vira uma linha do relatório, com extras replicados."""
    rows = wounds_to_rows("img.png", measure_wounds(two_wounds_mask), dice=0.9)

    assert len(rows) == 2
    assert [r["wound_id"] for r in rows] == [1, 2]
    assert all(r["dice"] == 0.9 for r in rows)
    assert all(r["nome_imagem"] == "img.png" for r in rows)


def test_format_text_without_scale_says_pixels(single_rect_mask):
    """O texto deixa explícito quando não há conversão física."""
    text = format_measurements_text(measure_wounds(single_rect_mask))
    measurement_lines = [
        line for line in text.splitlines()
        if line.startswith("  Área") or line.startswith("  Comprimento")
        or line.startswith("  Largura") or line.startswith("  Perímetro")
    ]

    assert "Ferida 1" in text
    assert measurement_lines
    # Nenhuma linha de medida apresenta unidade física.
    assert all("cm" not in line for line in measurement_lines)
    assert all("px" in line for line in measurement_lines)
    assert "Sem fator de escala" in text


def test_format_text_with_scale_shows_cm(single_rect_mask):
    """Com escala, os valores em cm aparecem junto aos pixels."""
    text = format_measurements_text(measure_wounds(single_rect_mask, pixels_por_cm=10.0))

    assert "cm²" in text
    assert "72.0 cm²" in text
    assert "Sem fator de escala" not in text


def test_format_text_when_no_wound(empty_mask):
    """Mensagem clara quando nada foi detectado."""
    assert format_measurements_text(measure_wounds(empty_mask)) == (
        "Nenhuma ferida detectada na imagem."
    )


# --------------------------------------------------------------------------- #
# 8. Visualização
# --------------------------------------------------------------------------- #

def test_annotate_measurements_returns_new_image(single_rect_mask):
    """A anotação devolve uma nova imagem, sem alterar a original."""
    image = np.full((*single_rect_mask.shape, 3), 30, dtype=np.uint8)
    original = image.copy()

    annotated = annotate_measurements(image, measure_wounds(single_rect_mask),
                                      mask=single_rect_mask)

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8
    assert np.array_equal(image, original)  # entrada intacta
    assert not np.array_equal(annotated, original)  # algo foi desenhado


def test_annotate_measurements_handles_no_wounds(empty_mask):
    """Sem ferida, a imagem anotada é apenas a imagem original."""
    image = np.full((*empty_mask.shape, 3), 30, dtype=np.uint8)
    annotated = annotate_measurements(image, [], mask=empty_mask)
    assert annotated.shape == image.shape


def test_annotate_wound_at_image_border():
    """Rótulo de ferida colada na borda não estoura os limites da imagem."""
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[0:60, 0:60] = 1
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    annotated = annotate_measurements(image, measure_wounds(mask), mask=mask)
    assert annotated.shape == (120, 120, 3)


def test_annotate_multiple_wounds(two_wounds_mask):
    """Todas as feridas são anotadas na mesma imagem."""
    image = np.full((*two_wounds_mask.shape, 3), 30, dtype=np.uint8)
    annotated = annotate_measurements(image, measure_wounds(two_wounds_mask),
                                      mask=two_wounds_mask)
    assert annotated.shape == image.shape


# --------------------------------------------------------------------------- #
# 9. Compatibilidade da estrutura de dados
# --------------------------------------------------------------------------- #

def test_wound_measurements_legacy_construction():
    """A dataclass ainda pode ser construída só com os campos originais."""
    m = WoundMeasurements(
        area_pixels=100,
        perimeter_pixels=40.0,
        bounding_box=(0, 0, 10, 10),
        centroid=(5.0, 5.0),
        circularity=0.78,
        aspect_ratio=1.0,
        convex_area_pixels=100.0,
        solidity=1.0,
        percent_image_area=1.0,
        num_connected_components=1,
    )

    assert m.wound_id == 1
    assert m.area_cm2 is None
    assert m.has_physical_units is False
