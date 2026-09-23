"""
test_pipeline_integration.py
============================
Testes de integração do fluxo completo:

    imagem -> segmentação -> máscara -> medição -> apresentação

Validam que a segmentação existente continua funcionando **e** que a
medição foi corretamente acoplada a ela, sem alterar o contrato dos
módulos já existentes.

Estes testes exigem a pilha de deep learning (``torch``,
``segmentation_models_pytorch``, ``albumentations``, ``matplotlib``).
Quando ela não está instalada, os testes são pulados — os testes de
medição em ``test_measure.py`` rodam apenas com ``opencv`` + ``numpy``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")
pytest.importorskip("segmentation_models_pytorch")

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # backend sem display, obrigatório em CI

import config  # noqa: E402
from models import build_model  # noqa: E402

IMAGE_H, IMAGE_W = 300, 400


class _StubSegmenter(torch.nn.Module):
    """Modelo de segmentação determinístico usado no lugar de um treinado.

    Devolve logits altos em um retângulo conhecido (em coordenadas
    relativas) e baixos no restante, de modo que a máscara prevista seja
    previsível e as medições possam ser conferidas analiticamente.
    """

    def __init__(self, box=(0.25, 0.25, 0.75, 0.5)) -> None:
        super().__init__()
        self.box = box

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        logits = torch.full((n, 1, h, w), -10.0)
        x0, y0, x1, y1 = self.box
        logits[:, :, int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 10.0
        return logits


@pytest.fixture
def sample_image(tmp_path):
    """Grava uma imagem RGB sintética em disco e devolve seu caminho."""
    import cv2

    image = np.full((IMAGE_H, IMAGE_W, 3), 60, dtype=np.uint8)
    image[80:200, 100:300] = (180, 90, 90)  # "ferida" avermelhada
    path = tmp_path / "ferida_teste.png"
    cv2.imwrite(str(path), image)
    return path


# --------------------------------------------------------------------------- #
# A segmentação continua funcionando
# --------------------------------------------------------------------------- #

def test_build_model_still_works():
    """A fábrica de modelos continua produzindo um segmentador válido.

    ``encoder_weights=None`` evita baixar os pesos ImageNet no teste.
    """
    from config import ModelConfig

    model = build_model(ModelConfig(encoder_weights=None))
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))

    assert out.shape == (1, 1, 64, 64)


def test_predict_mask_returns_mask_at_original_resolution(sample_image):
    """`predict_mask` devolve a máscara já reescalada ao tamanho original."""
    from predict import predict_mask

    model = _StubSegmenter()
    image, mask = predict_mask(model, sample_image, torch.device("cpu"))

    assert image.shape == (IMAGE_H, IMAGE_W, 3)
    assert mask.shape == (IMAGE_H, IMAGE_W)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.sum() > 0


def test_predicted_mask_matches_expected_region(sample_image):
    """A região segmentada corresponde ao retângulo previsto pelo modelo."""
    from predict import predict_mask

    _, mask = predict_mask(_StubSegmenter(), sample_image, torch.device("cpu"))

    ys, xs = np.nonzero(mask)
    assert xs.min() == pytest.approx(0.25 * IMAGE_W, abs=3)
    assert xs.max() == pytest.approx(0.75 * IMAGE_W, abs=3)
    assert ys.min() == pytest.approx(0.25 * IMAGE_H, abs=3)
    assert ys.max() == pytest.approx(0.50 * IMAGE_H, abs=3)


# --------------------------------------------------------------------------- #
# Segmentação -> medição
# --------------------------------------------------------------------------- #

def test_measurement_on_predicted_mask(sample_image):
    """A máscara prevista alimenta a medição e produz valores coerentes."""
    from measure import measure_wounds
    from predict import predict_mask

    _, mask = predict_mask(_StubSegmenter(), sample_image, torch.device("cpu"))
    wounds = measure_wounds(mask)

    assert len(wounds) == 1
    w = wounds[0]
    expected_w = 0.50 * IMAGE_W  # 25% -> 75%
    expected_h = 0.25 * IMAGE_H  # 25% -> 50%

    assert w.area_pixels == pytest.approx(expected_w * expected_h, rel=0.05)
    assert w.length_pixels == pytest.approx(expected_w, rel=0.05)
    assert w.width_pixels == pytest.approx(expected_h, rel=0.05)
    assert w.has_physical_units is False


def test_predict_single_image_full_pipeline(sample_image, tmp_path, monkeypatch):
    """Fluxo completo do `predict.py`: artefatos, CSV e medições."""
    import pandas as pd

    import predict as predict_module

    out_dir = tmp_path / "predictions"
    out_dir.mkdir()
    monkeypatch.setattr(config, "PREDICTIONS_DIR", out_dir)
    monkeypatch.setattr(
        predict_module, "load_model_from_checkpoint",
        lambda checkpoint_path, device: _StubSegmenter(),
    )

    result = predict_module.predict_single_image(
        image_path=sample_image,
        checkpoint_path=tmp_path / "inexistente.pt",  # não é lido (stub)
        pixels_por_cm=10.0,
    )

    stem = sample_image.stem
    assert (out_dir / f"{stem}_comparacao.png").exists()
    assert (out_dir / f"{stem}_overlay.png").exists()
    assert (out_dir / f"{stem}_medicoes.png").exists()

    csv_path = out_dir / f"{stem}_medicoes.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.loc[0, "wound_id"] == 1
    assert df.loc[0, "area_pixels"] > 0

    # Formato de retorno histórico preservado + novos campos
    assert result["num_feridas"] == 1
    assert result["area_pixels"] > 0
    assert result["area_cm2"] == pytest.approx(result["area_pixels"] / 100.0)
    assert len(result["feridas"]) == 1


def test_predict_single_image_without_wound(sample_image, tmp_path, monkeypatch):
    """Nenhuma ferida detectada: o pipeline não quebra e sinaliza o caso."""
    import predict as predict_module

    class _EmptySegmenter(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), -10.0)

    out_dir = tmp_path / "predictions_vazio"
    out_dir.mkdir()
    monkeypatch.setattr(config, "PREDICTIONS_DIR", out_dir)
    monkeypatch.setattr(
        predict_module, "load_model_from_checkpoint",
        lambda checkpoint_path, device: _EmptySegmenter(),
    )

    result = predict_module.predict_single_image(
        image_path=sample_image,
        checkpoint_path=tmp_path / "inexistente.pt",
    )

    assert result["num_feridas"] == 0
    assert result["feridas"][0]["observacao"] == "nenhuma ferida detectada"
    assert (out_dir / f"{sample_image.stem}_medicoes.png").exists()


def test_visualize_prediction_panels(sample_image, tmp_path):
    """O painel de medições é opcional — o grid antigo continua válido."""
    from measure import annotate_measurements, measure_wounds
    from predict import predict_mask, visualize_prediction

    image, mask = predict_mask(_StubSegmenter(), sample_image, torch.device("cpu"))

    legacy_path = tmp_path / "legacy.png"
    visualize_prediction(image, mask, save_path=legacy_path)
    assert legacy_path.exists()

    annotated = annotate_measurements(image, measure_wounds(mask), mask=mask)
    new_path = tmp_path / "com_medicoes.png"
    visualize_prediction(image, mask, ground_truth_mask=mask,
                        save_path=new_path, annotated=annotated)
    assert new_path.exists()


# --------------------------------------------------------------------------- #
# Avaliação no conjunto de teste
# --------------------------------------------------------------------------- #

def test_collect_predictions_returns_per_wound_records():
    """`evaluate.collect_predictions` passa a devolver também o detalhe por ferida."""
    pytest.importorskip("torchmetrics")
    pytest.importorskip("sklearn")
    from evaluate import collect_predictions

    device = torch.device("cpu")
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 10:40, 10:50] = 1.0
    images = torch.randn(2, 3, 64, 64)
    loader = [(images, masks, ["a.png", "b.png"])]

    y_true, y_prob, per_image, per_wound, agg = collect_predictions(
        _StubSegmenter(), loader, device
    )

    assert y_true.shape == y_prob.shape
    assert len(per_image) == 2
    assert len(per_wound) == 2  # uma ferida por imagem
    assert {r["nome_imagem"] for r in per_wound} == {"a.png", "b.png"}
    assert all("dice" in r for r in per_wound)
    assert all(r["num_feridas"] == 1 for r in per_image)
    assert set(agg) == {"iou", "precision", "recall", "f1", "accuracy"}


# --------------------------------------------------------------------------- #
# Inferência em lote
# --------------------------------------------------------------------------- #

def test_batch_inference_generates_both_reports(sample_image, tmp_path, monkeypatch):
    """O lote gera o CSV por imagem (legado) e o novo CSV por ferida."""
    import shutil

    import pandas as pd

    import batch_inference as batch_module

    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    for i in range(2):
        shutil.copy2(sample_image, input_dir / f"img_{i}.png")

    monkeypatch.setattr(
        batch_module, "load_model_from_checkpoint",
        lambda checkpoint_path, device: _StubSegmenter(),
    )

    out_dir = tmp_path / "lote"
    df = batch_module.run_batch_inference(
        input_dir=input_dir,
        checkpoint_path=tmp_path / "inexistente.pt",
        output_dir=out_dir,
    )

    assert len(df) == 2
    assert (df["num_feridas"] == 1).all()
    assert (out_dir / "medicoes_lote.csv").exists()

    wounds_df = pd.read_csv(out_dir / "medicoes_por_ferida.csv")
    assert len(wounds_df) == 2
    assert set(wounds_df["nome_imagem"]) == {"img_0.png", "img_1.png"}
    assert (out_dir / "img_0_medicoes.png").exists()
    assert (out_dir / "img_0_mask.png").exists()  # artefatos antigos preservados
    assert (out_dir / "img_0_overlay.png").exists()
