from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch

from train.model import build_multitask_model
import train.predict as predict_module
from train.predict import (
    _load_bootstrap_model,
    _iter_tile_bounds,
    _resolve_scene_roots,
    build_prediction_output_root,
    compute_probability_stats,
    export_five_band_predictions,
    build_head_masks,
    make_overlay_image,
    suppress_aging_near_paint_edges,
    parse_args,
    resolve_paint_pollution_conflict,
    resolve_paint_pollution_conflict_with_thresholds,
    threshold_prediction,
)


def test_threshold_prediction_returns_binary_uint8_mask() -> None:
    prediction = np.array([[0.1, 0.8], [0.7, 0.2]], dtype=np.float32)

    mask = threshold_prediction(prediction, threshold=0.5)

    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 255], [255, 0]]


def test_parse_args_accepts_threshold_override() -> None:
    args = parse_args(["--threshold", "0.2"])

    assert args.threshold == 0.2


def test_parse_args_accepts_per_head_threshold_overrides() -> None:
    args = parse_args(["--paint-threshold", "0.6", "--pollution-threshold", "0.35", "--aging-threshold", "0.4"])

    assert args.paint_threshold == 0.6
    assert args.pollution_threshold == 0.35
    assert args.aging_threshold == 0.4


def test_parse_args_accepts_probability_diagnostics_options() -> None:
    args = parse_args(["--diagnostic-csv", "C:/tmp/stats.csv", "--save-aging-probability-map"])

    assert args.diagnostic_csv == "C:/tmp/stats.csv"
    assert args.save_aging_probability_map is True


def test_parse_args_accepts_export_heads() -> None:
    args = parse_args(["--export-heads", "aging"])

    assert args.export_heads == ["aging"]


def test_parse_args_accepts_composition_mode() -> None:
    args = parse_args(["--composition-mode", "independent"])

    assert args.composition_mode == "independent"


def test_compute_probability_stats_reports_gt_positive_distribution() -> None:
    probabilities = np.array([[0.1, 0.6], [0.8, 0.4]], dtype=np.float32)
    gt_mask = np.array([[0, 255], [0, 255]], dtype=np.uint8)

    stats = compute_probability_stats(probabilities, gt_mask=gt_mask)

    assert np.isclose(stats["max"], 0.8)
    assert stats["p95"] > stats["mean"]
    assert stats["gt_positive_pixels"] == 2
    assert stats["positive_gt_mean"] == 0.5


def test_iter_tile_bounds_covers_trailing_edge() -> None:
    assert _iter_tile_bounds(length=1100, tile_size=512, tile_stride=512) == [
        (0, 512),
        (512, 1024),
        (588, 1100),
    ]


def test_resolve_paint_pollution_conflict_keeps_only_higher_probability_head() -> None:
    paint = np.array([[0.9, 0.4], [0.8, 0.1]], dtype=np.float32)
    pollution = np.array([[0.8, 0.7], [0.2, 0.6]], dtype=np.float32)

    paint_mask, pollution_mask = resolve_paint_pollution_conflict(paint, pollution, threshold=0.5)

    assert paint_mask.tolist() == [[255, 0], [255, 0]]
    assert pollution_mask.tolist() == [[0, 255], [0, 255]]


def test_resolve_paint_pollution_conflict_with_thresholds_respects_per_head_activation() -> None:
    paint = np.array([[0.45, 0.8]], dtype=np.float32)
    pollution = np.array([[0.45, 0.7]], dtype=np.float32)

    paint_mask, pollution_mask = resolve_paint_pollution_conflict_with_thresholds(
        paint,
        pollution,
        paint_threshold=0.6,
        pollution_threshold=0.4,
    )

    assert paint_mask.tolist() == [[0, 255]]
    assert pollution_mask.tolist() == [[255, 0]]


def test_make_overlay_image_applies_color_on_positive_mask() -> None:
    preview = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    overlay = make_overlay_image(preview, mask, color=(255, 0, 0))

    assert overlay.shape == (4, 4, 3)
    assert overlay[0, 0].tolist() == [0, 0, 0]
    assert overlay[1, 1][0] > 0


def test_build_head_masks_independent_keeps_overlapping_paint_and_pollution() -> None:
    probabilities = {
        "paint": np.array([[0.9, 0.2]], dtype=np.float32),
        "pollution": np.array([[0.8, 0.7]], dtype=np.float32),
        "aging": np.array([[0.1, 0.9]], dtype=np.float32),
    }

    masks = build_head_masks(
        probabilities,
        paint_threshold=0.5,
        pollution_threshold=0.5,
        aging_threshold=0.5,
        composition_mode="independent",
    )

    assert masks["paint"].tolist() == [[255, 0]]
    assert masks["pollution"].tolist() == [[255, 255]]
    assert masks["aging"].tolist() == [[0, 255]]


def test_build_head_masks_conflict_resolved_splits_overlapping_paint_and_pollution() -> None:
    probabilities = {
        "paint": np.array([[0.9, 0.2]], dtype=np.float32),
        "pollution": np.array([[0.8, 0.7]], dtype=np.float32),
        "aging": np.array([[0.1, 0.9]], dtype=np.float32),
    }

    masks = build_head_masks(
        probabilities,
        paint_threshold=0.5,
        pollution_threshold=0.5,
        aging_threshold=0.5,
        composition_mode="conflict_resolved",
    )

    assert masks["paint"].tolist() == [[255, 0]]
    assert masks["pollution"].tolist() == [[0, 255]]
    assert masks["aging"].tolist() == [[0, 255]]



def test_suppress_aging_near_paint_edges_removes_weak_edge_response_but_keeps_far_pixels() -> None:
    aging_probability = np.array(
        [
            [0.0, 0.62, 0.62, 0.62, 0.0, 0.0],
            [0.0, 0.62, 0.20, 0.62, 0.0, 0.0],
            [0.0, 0.62, 0.20, 0.62, 0.0, 0.72],
            [0.0, 0.62, 0.62, 0.62, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    paint_mask = np.array(
        [
            [0, 255, 255, 255, 0, 0],
            [0, 255, 255, 255, 0, 0],
            [0, 255, 255, 255, 0, 0],
            [0, 255, 255, 255, 0, 0],
        ],
        dtype=np.uint8,
    )

    suppressed = suppress_aging_near_paint_edges(
        aging_probability,
        paint_mask,
        aging_threshold=0.5,
        edge_width=1,
        keep_threshold=0.8,
    )

    assert suppressed.tolist() == [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 255],
        [0, 0, 0, 0, 0, 0],
    ]


def test_suppress_aging_near_paint_edges_keeps_strong_edge_response() -> None:
    aging_probability = np.array([[0.0, 0.93, 0.0]], dtype=np.float32)
    paint_mask = np.array([[0, 255, 255]], dtype=np.uint8)

    suppressed = suppress_aging_near_paint_edges(
        aging_probability,
        paint_mask,
        aging_threshold=0.5,
        edge_width=1,
        keep_threshold=0.8,
    )

    assert suppressed.tolist() == [[0, 255, 0]]


def test_build_head_masks_applies_aging_edge_suppression_by_default() -> None:
    probabilities = {
        "paint": np.array([[0.9, 0.9, 0.0]], dtype=np.float32),
        "pollution": np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        "aging": np.array([[0.62, 0.62, 0.85]], dtype=np.float32),
    }

    masks = build_head_masks(
        probabilities,
        paint_threshold=0.5,
        pollution_threshold=0.5,
        aging_threshold=0.5,
        composition_mode="conflict_resolved",
    )

    assert masks["aging"].tolist() == [[0, 0, 255]]

def test_filter_pollution_components_removes_large_smooth_component() -> None:
    roi_preview = np.full((64, 64, 3), 140, dtype=np.uint8)
    roi_mask = np.zeros((64, 64), dtype=np.uint8)
    roi_mask[4:60, 8:56] = 255
    roi_probability = np.full((64, 64), 0.9, dtype=np.float32)

    filtered = predict_module.filter_pollution_components(
        roi_mask,
        roi_probability,
        roi_preview,
    )

    assert int((filtered > 0).sum()) == 0


def test_filter_pollution_components_keeps_granular_irregular_cluster() -> None:
    roi_preview = np.full((64, 64, 3), 130, dtype=np.uint8)
    roi_mask = np.zeros((64, 64), dtype=np.uint8)
    coords = [
        (20, 20), (21, 20), (22, 20), (20, 21), (22, 21), (20, 22), (21, 22), (22, 22),
        (25, 24), (26, 24), (24, 25), (26, 25), (24, 26), (25, 26),
        (29, 28), (30, 28), (28, 29), (30, 29), (28, 30), (29, 30),
    ]
    for x, y in coords:
        roi_mask[y, x] = 255
        roi_preview[y, x] = np.array([210, 190, 70], dtype=np.uint8)
    roi_probability = np.where(roi_mask > 0, 0.85, 0.05).astype(np.float32)

    filtered = predict_module.filter_pollution_components(
        roi_mask,
        roi_probability,
        roi_preview,
    )

    assert int((filtered > 0).sum()) == int((roi_mask > 0).sum())


def test_parse_args_accepts_pollution_shape_filter_options() -> None:
    args = parse_args([
        "--pollution-shape-filter",
        "--pollution-max-smooth-area-ratio",
        "0.12",
    ])

    assert args.pollution_shape_filter is True
    assert args.pollution_max_smooth_area_ratio == 0.12


def test_parse_args_accepts_pollution_core_threshold_option() -> None:
    args = parse_args([
        "--pollution-shape-filter",
        "--pollution-core-threshold",
        "0.78",
    ])

    assert args.pollution_shape_filter is True
    assert args.pollution_core_threshold == 0.78


def test_prediction_output_directory_is_separate() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_root = project_root / "train" / "experiments" / "five_band_predictions"
    sentinel_root = project_root / "train" / "five_band_patches"

    assert output_root != sentinel_root


def test_prediction_output_directory_includes_variant_and_epoch() -> None:
    output_root = build_prediction_output_root(
        project_root=Path("C:/demo"),
        model_variant="attention",
        epochs=3,
    )

    assert output_root == Path("C:/demo/train/experiments/five_band_predictions/attention/epochs_3")


def test_load_bootstrap_model_uses_checkpoint_variant(tmp_path: Path) -> None:
    model = build_multitask_model(
        variant="attention",
        encoder_name="resnet18",
        in_channels=3,
        head_names=("paint", "pollution", "aging"),
    )
    checkpoint_path = tmp_path / "attention.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_variant": "attention",
            "image_size": 128,
        },
        checkpoint_path,
    )

    loaded_model, checkpoint = _load_bootstrap_model(checkpoint_path, torch.device("cpu"))

    assert checkpoint["model_variant"] == "attention"
    assert loaded_model.__class__.__name__ == model.__class__.__name__


def test_load_bootstrap_model_supports_task_specific_variant(tmp_path: Path) -> None:
    model = build_multitask_model(
        variant="task_specific",
        encoder_name="resnet18",
        in_channels=5,
        head_names=("paint", "pollution", "aging"),
    )
    checkpoint_path = tmp_path / "task_specific.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_variant": "task_specific",
            "in_channels": 5,
            "image_size": 128,
        },
        checkpoint_path,
    )

    loaded_model, checkpoint = _load_bootstrap_model(checkpoint_path, torch.device("cpu"))

    assert checkpoint["model_variant"] == "task_specific"
    assert checkpoint["in_channels"] == 5
    assert loaded_model.__class__.__name__ == model.__class__.__name__


def test_load_bootstrap_model_supports_pigment_aux_head(tmp_path: Path) -> None:
    model = build_multitask_model(
        variant="task_specific",
        encoder_name="resnet18",
        in_channels=15,
        head_names=("paint", "pollution", "aging"),
        pigment_class_count=7,
    )
    checkpoint_path = tmp_path / "task_specific_pigment.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_variant": "task_specific",
            "in_channels": 15,
            "image_size": 128,
            "pigment_class_names": (
                "无颜料",
                "石绿",
                "石青",
                "朱砂",
                "代赭",
                "石青+朱砂",
                "石青+代赭+朱砂",
            ),
        },
        checkpoint_path,
    )

    loaded_model, checkpoint = _load_bootstrap_model(checkpoint_path, torch.device("cpu"))

    assert checkpoint["pigment_class_names"][0] == "无颜料"
    assert set(loaded_model(torch.randn(1, 15, 64, 64))) == {"paint", "pollution", "aging", "pigment"}

def test_build_pigment_summary_includes_margin_and_confidence_metadata() -> None:
    clear_summary = predict_module._build_pigment_summary(
        np.asarray([3.0, 2.3, 0.0], dtype=np.float32),
        ("class_a", "class_b", "class_c"),
        paint_positive_pixels=320,
        paint_total_pixels=4096,
    )
    close_summary = predict_module._build_pigment_summary(
        np.asarray([3.0, 2.7, 0.0], dtype=np.float32),
        ("class_a", "class_b", "class_c"),
        paint_positive_pixels=320,
        paint_total_pixels=4096,
    )
    small_region_summary = predict_module._build_pigment_summary(
        np.asarray([4.0, 1.0, 0.0], dtype=np.float32),
        ("class_a", "class_b", "class_c"),
        paint_positive_pixels=12,
        paint_total_pixels=4096,
    )

    assert clear_summary["predicted_label"] == "class_a"
    assert clear_summary["margin"] > close_summary["margin"] > 0
    assert clear_summary["confidence_tier"] == "clear"
    assert clear_summary["low_confidence"] is False
    assert close_summary["confidence_tier"] == "close"
    assert close_summary["low_confidence"] is False
    assert small_region_summary["confidence_tier"] == "review"
    assert small_region_summary["low_confidence"] is True
    assert isinstance(clear_summary["review_reason"], str) and clear_summary["review_reason"]
    assert isinstance(close_summary["review_reason"], str) and close_summary["review_reason"]
    assert isinstance(small_region_summary["review_reason"], str) and small_region_summary["review_reason"]


def test_build_pigment_summary_small_region_review_is_not_scaled_by_full_image_size() -> None:
    compact_canvas_summary = predict_module._build_pigment_summary(
        np.asarray([4.0, 1.0, 0.0], dtype=np.float32),
        ("class_a", "class_b", "class_c"),
        paint_positive_pixels=80,
        paint_total_pixels=4096,
    )
    large_canvas_summary = predict_module._build_pigment_summary(
        np.asarray([4.0, 1.0, 0.0], dtype=np.float32),
        ("class_a", "class_b", "class_c"),
        paint_positive_pixels=80,
        paint_total_pixels=65536,
    )

    assert compact_canvas_summary["margin"] == large_canvas_summary["margin"]
    assert compact_canvas_summary["confidence_tier"] == large_canvas_summary["confidence_tier"] == "clear"
    assert compact_canvas_summary["low_confidence"] is False
    assert large_canvas_summary["low_confidence"] is False
    assert isinstance(compact_canvas_summary["review_reason"], str) and compact_canvas_summary["review_reason"]
    assert isinstance(large_canvas_summary["review_reason"], str) and large_canvas_summary["review_reason"]


def test_build_pigment_summary_small_region_review_uses_fixed_pixel_floor() -> None:
    review_summary = predict_module._build_pigment_summary(
        np.asarray([4.0, 1.0, 0.0], dtype=np.float32),
        ("class_a", "class_b", "class_c"),
        paint_positive_pixels=63,
        paint_total_pixels=65536,
    )
    clear_summary = predict_module._build_pigment_summary(
        np.asarray([4.0, 1.0, 0.0], dtype=np.float32),
        ("class_a", "class_b", "class_c"),
        paint_positive_pixels=64,
        paint_total_pixels=65536,
    )

    assert review_summary["margin"] == clear_summary["margin"]
    assert review_summary["confidence_tier"] == "review"
    assert review_summary["low_confidence"] is True
    assert clear_summary["confidence_tier"] == "clear"
    assert clear_summary["low_confidence"] is False
    assert isinstance(review_summary["review_reason"], str) and review_summary["review_reason"]
    assert isinstance(clear_summary["review_reason"], str) and clear_summary["review_reason"]
def test_export_five_band_predictions_rejects_non_five_channel_checkpoint(tmp_path: Path) -> None:
    model = build_multitask_model(
        variant="baseline",
        encoder_name="resnet18",
        in_channels=3,
        head_names=("paint", "pollution", "aging"),
    )
    checkpoint_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_variant": "baseline",
            "in_channels": 3,
            "image_size": 128,
            "epochs": 1,
        },
        checkpoint_path,
    )
    scenes_root = tmp_path / "scenes"
    scenes_root.mkdir()

    try:
        export_five_band_predictions(checkpoint_path, scenes_root, tmp_path / "predictions")
    except ValueError as exc:
        assert "five-band training pipeline" in str(exc)
    else:
        raise AssertionError("Expected export_five_band_predictions to reject non-five-channel checkpoints.")


def test_resolve_scene_roots_filters_explicit_scene_ids(tmp_path: Path) -> None:
    wanted = tmp_path / "SAMPLE_001"
    wanted.mkdir()
    np.save(wanted / "five_band.npy", np.zeros((4, 4, 5), dtype=np.float32))
    extra = tmp_path / "CAMERA_001"
    extra.mkdir()
    np.save(extra / "five_band.npy", np.zeros((4, 4, 5), dtype=np.float32))

    resolved = _resolve_scene_roots(tmp_path, ["SAMPLE_001"])

    assert resolved == [wanted]


def test_export_five_band_predictions_limits_masks_to_board_roi(tmp_path: Path, monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = tensor.shape[0]
            full = torch.full((batch, 1, 4, 4), 10.0)
            return {
                "paint": full,
                "pollution": full,
                "aging": full,
                "pigment": torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32).repeat(batch, 1),
            }

    def fake_load_bootstrap_model(_checkpoint_path: Path, _device: torch.device):
        return DummyModel(), {
            "in_channels": 15,
            "image_size": 4,
            "model_variant": "task_specific",
            "epochs": 3,
            "pigment_class_names": ("无颜料", "石青", "石青+朱砂"),
        }

    monkeypatch.setattr(predict_module, "_load_bootstrap_model", fake_load_bootstrap_model)

    scene_root = tmp_path / "scenes" / "SAMPLE_001"
    scene_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.zeros((8, 12, 5), dtype=np.float32))

    preview = np.zeros((8, 12, 3), dtype=np.uint8)
    preview[:, :] = np.array([240, 120, 240], dtype=np.uint8)
    preview[1:7, 2:10] = np.array([40, 160, 90], dtype=np.uint8)
    Image.fromarray(preview).save(scene_root / "preview.png")

    exported = export_five_band_predictions(
        checkpoint_path=tmp_path / "dummy.pt",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "predictions",
        scene_ids=["SAMPLE_001"],
        threshold=0.5,
    )

    assert len(exported) == 8
    mask = np.asarray(Image.open(tmp_path / "predictions" / "SAMPLE_001" / "paint_pred.png").convert("L"))
    assert mask[0, 0] == 0
    assert mask[7, 11] == 0
    assert mask[2, 3] == 255
    assert (tmp_path / "predictions" / "SAMPLE_001" / "pigment_summary.json").exists()


def test_export_five_band_predictions_uses_separate_head_thresholds(tmp_path: Path, monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = tensor.shape[0]
            paint = torch.full((batch, 1, 4, 4), -0.2, dtype=torch.float32)
            pollution = torch.full((batch, 1, 4, 4), -0.2, dtype=torch.float32)
            aging = torch.full((batch, 1, 4, 4), -0.2, dtype=torch.float32)
            return {
                "paint": paint,
                "pollution": pollution,
                "aging": aging,
            }

    def fake_load_bootstrap_model(_checkpoint_path: Path, _device: torch.device):
        return DummyModel(), {"in_channels": 15, "image_size": 4, "model_variant": "task_specific", "epochs": 3}

    monkeypatch.setattr(predict_module, "_load_bootstrap_model", fake_load_bootstrap_model)

    scene_root = tmp_path / "scenes" / "SAMPLE_001"
    scene_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.ones((8, 12, 5), dtype=np.float32))

    preview = np.zeros((8, 12, 3), dtype=np.uint8)
    preview[:, :] = np.array([240, 120, 240], dtype=np.uint8)
    preview[1:7, 2:10] = np.array([40, 160, 90], dtype=np.uint8)
    Image.fromarray(preview).save(scene_root / "preview.png")

    export_five_band_predictions(
        checkpoint_path=tmp_path / "dummy.pt",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "predictions",
        scene_ids=["SAMPLE_001"],
        paint_threshold=0.6,
        pollution_threshold=0.4,
        aging_threshold=0.6,
    )

    paint = np.asarray(Image.open(tmp_path / "predictions" / "SAMPLE_001" / "paint_pred.png").convert("L"))
    pollution = np.asarray(Image.open(tmp_path / "predictions" / "SAMPLE_001" / "pollution_pred.png").convert("L"))
    aging = np.asarray(Image.open(tmp_path / "predictions" / "SAMPLE_001" / "aging_pred.png").convert("L"))

    assert int((paint > 0).sum()) == 0
    assert int((pollution > 0).sum()) > 0
    assert int((aging > 0).sum()) == 0


def test_export_five_band_predictions_writes_aging_probability_diagnostics(tmp_path: Path, monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = tensor.shape[0]
            zeros = torch.zeros((batch, 1, 4, 4), dtype=torch.float32)
            aging = torch.tensor(
                [[[[0.0, 2.0, -2.0, 0.0], [0.0, 2.0, -2.0, 0.0], [0.0, 2.0, -2.0, 0.0], [0.0, 2.0, -2.0, 0.0]]]],
                dtype=torch.float32,
            ).repeat(batch, 1, 1, 1)
            return {
                "paint": zeros,
                "pollution": zeros,
                "aging": aging,
            }

    def fake_load_bootstrap_model(_checkpoint_path: Path, _device: torch.device):
        return DummyModel(), {"in_channels": 15, "image_size": 4, "model_variant": "task_specific", "epochs": 3}

    monkeypatch.setattr(predict_module, "_load_bootstrap_model", fake_load_bootstrap_model)

    scene_root = tmp_path / "scenes" / "SAMPLE_024"
    scene_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.ones((8, 12, 5), dtype=np.float32))

    preview = np.zeros((8, 12, 3), dtype=np.uint8)
    preview[:, :] = np.array([240, 120, 240], dtype=np.uint8)
    preview[1:7, 2:10] = np.array([40, 160, 90], dtype=np.uint8)
    Image.fromarray(preview).save(scene_root / "preview.png")
    masks_root = scene_root / "masks"
    masks_root.mkdir()
    aging_gt = np.zeros((8, 12), dtype=np.uint8)
    aging_gt[2:4, 3:5] = 255
    Image.fromarray(aging_gt).save(masks_root / "aging.png")

    diagnostic_csv = tmp_path / "predictions" / "aging_probability_stats.csv"
    export_five_band_predictions(
        checkpoint_path=tmp_path / "dummy.pt",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "predictions",
        scene_ids=["SAMPLE_024"],
        diagnostic_csv=diagnostic_csv,
        save_aging_probability_map=True,
    )

    with diagnostic_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["scene"] == "SAMPLE_024"
    assert rows[0]["head"] == "aging"
    assert float(rows[0]["max"]) > 0.75
    assert int(rows[0]["gt_positive_pixels"]) == 4
    assert (tmp_path / "predictions" / "SAMPLE_024" / "aging_probability.png").exists()


def test_export_five_band_predictions_uses_augmented_fifteen_channel_input(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            captured["channels"] = int(tensor.shape[1])
            batch = tensor.shape[0]
            zeros = torch.zeros((batch, 1, 4, 4), dtype=torch.float32)
            return {
                "paint": zeros,
                "pollution": zeros,
                "aging": zeros,
            }

    def fake_load_bootstrap_model(_checkpoint_path: Path, _device: torch.device):
        return DummyModel(), {"in_channels": 15, "image_size": 4, "model_variant": "task_specific", "epochs": 3}

    monkeypatch.setattr(predict_module, "_load_bootstrap_model", fake_load_bootstrap_model)

    scene_root = tmp_path / "scenes" / "SAMPLE_001"
    scene_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.ones((8, 12, 5), dtype=np.float32))

    preview = np.zeros((8, 12, 3), dtype=np.uint8)
    preview[:, :] = np.array([240, 120, 240], dtype=np.uint8)
    preview[1:7, 2:10] = np.array([40, 160, 90], dtype=np.uint8)
    Image.fromarray(preview).save(scene_root / "preview.png")

    export_five_band_predictions(
        checkpoint_path=tmp_path / "dummy.pt",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "predictions",
        scene_ids=["SAMPLE_001"],
        threshold=0.5,
    )

    assert captured["channels"] == 15







def test_export_five_band_predictions_independent_mode_keeps_overlapping_masks(tmp_path: Path, monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = tensor.shape[0]
            paint = torch.tensor([[[[2.0, -2.0], [-2.0, -2.0]]]], dtype=torch.float32).repeat(batch, 1, 1, 1)
            pollution = torch.tensor([[[[2.0, 2.0], [-2.0, -2.0]]]], dtype=torch.float32).repeat(batch, 1, 1, 1)
            aging = torch.tensor([[[[-2.0, -2.0], [2.0, -2.0]]]], dtype=torch.float32).repeat(batch, 1, 1, 1)
            return {
                "paint": paint,
                "pollution": pollution,
                "aging": aging,
            }

    def fake_load_bootstrap_model(_checkpoint_path: Path, _device: torch.device):
        return DummyModel(), {"in_channels": 15, "image_size": 2, "model_variant": "task_specific", "epochs": 3}

    monkeypatch.setattr(predict_module, "_load_bootstrap_model", fake_load_bootstrap_model)

    scene_root = tmp_path / "scenes" / "SAMPLE_044"
    scene_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.ones((4, 4, 5), dtype=np.float32))

    preview = np.zeros((4, 4, 3), dtype=np.uint8)
    preview[:, :] = np.array([40, 160, 90], dtype=np.uint8)
    Image.fromarray(preview).save(scene_root / "preview.png")

    export_five_band_predictions(
        checkpoint_path=tmp_path / "dummy.pt",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "predictions",
        scene_ids=["SAMPLE_044"],
        threshold=0.5,
        composition_mode="independent",
    )

    sample_root = tmp_path / "predictions" / "SAMPLE_044"
    paint = np.asarray(Image.open(sample_root / "paint_pred.png").convert("L"))
    pollution = np.asarray(Image.open(sample_root / "pollution_pred.png").convert("L"))
    aging = np.asarray(Image.open(sample_root / "aging_pred.png").convert("L"))

    assert int((paint > 0).sum()) > 0
    assert int((pollution > 0).sum()) > 0
    assert int(((paint > 0) & (pollution > 0)).sum()) > 0
    assert int((aging > 0).sum()) > 0
    assert (sample_root / "combined_overlay.png").exists()


def test_export_five_band_predictions_can_export_only_selected_heads(tmp_path: Path, monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = tensor.shape[0]
            ones = torch.full((batch, 1, 4, 4), 1.0, dtype=torch.float32)
            return {
                "paint": ones,
                "pollution": ones,
                "aging": ones,
                "pigment": torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32).repeat(batch, 1),
            }

    def fake_load_bootstrap_model(_checkpoint_path: Path, _device: torch.device):
        return DummyModel(), {
            "in_channels": 15,
            "image_size": 4,
            "model_variant": "task_specific",
            "epochs": 3,
            "pigment_class_names": ("???", "??", "??+??"),
        }

    monkeypatch.setattr(predict_module, "_load_bootstrap_model", fake_load_bootstrap_model)

    scene_root = tmp_path / "scenes" / "SAMPLE_041"
    scene_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.ones((8, 12, 5), dtype=np.float32))

    preview = np.zeros((8, 12, 3), dtype=np.uint8)
    preview[:, :] = np.array([240, 120, 240], dtype=np.uint8)
    preview[1:7, 2:10] = np.array([40, 160, 90], dtype=np.uint8)
    Image.fromarray(preview).save(scene_root / "preview.png")

    exported = export_five_band_predictions(
        checkpoint_path=tmp_path / "dummy.pt",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "predictions",
        scene_ids=["SAMPLE_041"],
        export_heads=("aging",),
    )

    target_dir = tmp_path / "predictions" / "SAMPLE_041"
    assert (target_dir / "aging_pred.png").exists()
    assert (target_dir / "aging_overlay.png").exists()
    assert not (target_dir / "paint_pred.png").exists()
    assert not (target_dir / "pollution_pred.png").exists()
    assert not (target_dir / "combined_overlay.png").exists()
    assert not (target_dir / "pigment_summary.json").exists()
    assert len(exported) == 2
