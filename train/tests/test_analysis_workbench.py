import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from train.analysis_workbench import (
    HEADS,
    _compute_composite_pigment_score,
    _compute_curve_shape_score,
    _compute_peak_match_score,
    _compute_slope_score,
    compute_head_curve,
    export_workbench_manifest,
    normalize_curve,
)


def _write_scene(scene_root: Path) -> None:
    scene_root.mkdir(parents=True)
    preview = np.zeros((2, 3, 3), dtype=np.uint8)
    preview[:, :] = [30, 40, 50]
    Image.fromarray(preview).save(scene_root / "preview.png")
    cube = np.zeros((2, 3, 5), dtype=np.float32)
    for band_index in range(5):
        cube[:, :, band_index] = float(band_index + 1)
    np.save(scene_root / "five_band.npy", cube)


def _write_predictions(prediction_root: Path, scene_id: str) -> None:
    sample_root = prediction_root / scene_id
    sample_root.mkdir(parents=True)
    masks = {
        "paint": np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8),
        "pollution": np.array([[0, 255, 0], [0, 255, 0]], dtype=np.uint8),
        "aging": np.array([[0, 0, 255], [0, 0, 255]], dtype=np.uint8),
    }
    for head_name, mask in masks.items():
        Image.fromarray(mask).save(sample_root / f"{head_name}_pred.png")
        Image.fromarray(np.dstack([mask, mask, mask])).save(sample_root / f"{head_name}_overlay.png")
    Image.fromarray(np.full((2, 3, 3), 120, dtype=np.uint8)).save(sample_root / "combined_overlay.png")


def test_compute_head_curve_returns_region_mean_area_ratio_and_peak() -> None:
    five_band = np.zeros((2, 3, 5), dtype=np.float32)
    five_band[:, :, 0] = np.array([[1, 10, 10], [3, 10, 10]], dtype=np.float32)
    five_band[:, :, 1] = np.array([[2, 10, 10], [4, 10, 10]], dtype=np.float32)
    five_band[:, :, 2] = np.array([[8, 10, 10], [6, 10, 10]], dtype=np.float32)
    five_band[:, :, 3] = np.array([[4, 10, 10], [2, 10, 10]], dtype=np.float32)
    five_band[:, :, 4] = np.array([[5, 10, 10], [1, 10, 10]], dtype=np.float32)
    mask = np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)

    curve = compute_head_curve(five_band, mask)

    assert curve["values"] == [2.0, 3.0, 7.0, 3.0, 3.0]
    assert curve["normalized"] == [2.0 / 7.0, 3.0 / 7.0, 1.0, 3.0 / 7.0, 3.0 / 7.0]
    assert curve["area_ratio"] == 2 / 6
    assert curve["positive_pixels"] == 2
    assert curve["total_pixels"] == 6
    assert curve["peak_wavelength"] == 600
    assert curve["peak_value"] == 7.0


def test_normalize_curve_returns_zeroes_for_empty_or_flat_zero_curve() -> None:
    assert normalize_curve([]) == []
    assert normalize_curve([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_pigment_score_helpers_return_zero_for_empty_or_mismatched_curves() -> None:
    assert _compute_curve_shape_score([], [1.0, 0.5]) == 0.0
    assert _compute_peak_match_score([1.0, 0.5], [1.0]) == 0.0
    assert _compute_slope_score([1.0], [1.0, 0.5]) == 0.0
    composite_score, breakdown = _compute_composite_pigment_score([], [1.0, 0.5])
    assert composite_score == 0.0
    assert breakdown == {
        "curve_shape_score": 0.0,
        "peak_match_score": 0.0,
        "slope_score": 0.0,
    }


def test_peak_match_score_is_stable_for_plateau_peaks() -> None:
    plateau_curve = [0.30, 0.98, 0.98, 0.40, 0.10]
    shifted_plateau_curve = [0.28, 0.97, 0.98, 0.41, 0.12]
    distant_peak_curve = [0.99, 0.70, 0.42, 0.20, 0.08]

    stable_score = _compute_peak_match_score(plateau_curve, shifted_plateau_curve)
    distant_score = _compute_peak_match_score(plateau_curve, distant_peak_curve)

    assert stable_score == 1.0
    assert distant_score < stable_score


def test_slope_score_drops_for_opposite_trends() -> None:
    rising_curve = [0.10, 0.25, 0.50, 0.75, 0.95]
    falling_curve = [0.95, 0.75, 0.50, 0.25, 0.10]
    nearby_curve = [0.12, 0.24, 0.49, 0.72, 0.92]

    opposite_score = _compute_slope_score(rising_curve, falling_curve)
    nearby_score = _compute_slope_score(rising_curve, nearby_curve)

    assert opposite_score < nearby_score
    assert opposite_score < 0.7


def test_export_workbench_manifest_includes_pigment_candidates(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"

    for scene_id, curve in {
        "SAMPLE_001": [10.0, 20.0, 30.0, 40.0, 50.0],
        "SAMPLE_004": [50.0, 40.0, 30.0, 20.0, 10.0],
        "SAMPLE_017": [12.0, 19.0, 31.0, 39.0, 52.0],
    }.items():
        _write_scene(scenes_root / scene_id)
        cube = np.tile(np.asarray(curve, dtype=np.float32), (2, 3, 1))
        np.save(scenes_root / scene_id / "five_band.npy", cube)
        (scenes_root / scene_id / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)).save(
            scenes_root / scene_id / "masks" / "paint.png"
        )
        _write_predictions(prediction_root, scene_id)

    sample_record = tmp_path / "sample_record.md"
    sample_record.write_text(
        "# ??????\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | green |\n| SAMPLE_004 | blue |\n| SAMPLE_017 | green+red |\n",
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_017"],
        sample_record_path=sample_record,
        reference_scene_ids=["SAMPLE_001", "SAMPLE_004"],
    )

    pigment = manifest["samples"][0]["pigment_analysis"]
    assert pigment["enabled"] is True
    assert pigment["region_positive_pixels"] == 2
    assert pigment["top_candidates"][0]["name"] == "green"
    assert pigment["top_candidates"][0]["reference_sample_count"] == 1
    assert len(pigment["top_candidates"]) == 2
    assert pigment["cluster_analysis"]["available"] is True
    assert pigment["cluster_analysis"]["overlay"].endswith("dual_pigment_overlay.png")
    assert pigment["cluster_analysis"]["label_map"].endswith("dual_pigment_labels.png")
    assert pigment["cluster_analysis"]["summary_json"].endswith("dual_pigment_summary.json")
    mixed = manifest["samples"][0]["mixed_pigment_analysis"]
    assert mixed["enabled"] is True
    assert mixed["triggered"] is False



def test_export_workbench_manifest_uses_composite_pigment_score_breakdown(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"

    for scene_id, curve in {
        "SAMPLE_A": [0.20, 0.30, 0.90, 0.40, 0.20],
        "SAMPLE_B": [0.85, 0.72, 0.38, 0.18, 0.08],
        "TARGET": [0.22, 0.28, 0.88, 0.42, 0.21],
    }.items():
        _write_scene(scenes_root / scene_id)
        cube = np.tile(np.asarray(curve, dtype=np.float32), (2, 3, 1))
        np.save(scenes_root / scene_id / "five_band.npy", cube)
        (scenes_root / scene_id / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)).save(
            scenes_root / scene_id / "masks" / "paint.png"
        )
        _write_predictions(prediction_root, scene_id)

    sample_record = tmp_path / "sample_record.md"
    sample_record.write_text(
        "# sample records\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_A | azurite |\n| SAMPLE_B | malachite |\n| TARGET | azurite+vermilion |\n",
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["TARGET"],
        sample_record_path=sample_record,
        reference_scene_ids=["SAMPLE_A", "SAMPLE_B"],
    )

    pigment = manifest["samples"][0]["pigment_analysis"]
    assert pigment["top_candidates"][0]["name"] == "azurite"
    assert pigment["top_candidates"][0]["score"] > pigment["top_candidates"][1]["score"]

    breakdown = pigment["top_candidates"][0]["score_breakdown"]
    assert set(breakdown) == {"curve_shape_score", "peak_match_score", "slope_score"}
    assert breakdown["curve_shape_score"] > 0
    assert breakdown["peak_match_score"] > 0
    assert breakdown["slope_score"] > 0

def test_export_workbench_manifest_marks_close_candidates_for_review(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"

    curves = {
        "SAMPLE_001": [10.0, 20.0, 30.0, 40.0, 50.0],
        "SAMPLE_004": [11.0, 21.0, 29.0, 39.0, 49.0],
        "SAMPLE_018": [10.5, 20.5, 29.5, 39.5, 49.5],
    }
    for scene_id, curve in curves.items():
        _write_scene(scenes_root / scene_id)
        cube = np.tile(np.asarray(curve, dtype=np.float32), (2, 3, 1))
        np.save(scenes_root / scene_id / "five_band.npy", cube)
        (scenes_root / scene_id / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)).save(
            scenes_root / scene_id / "masks" / "paint.png"
        )
        _write_predictions(prediction_root, scene_id)

    sample_record = tmp_path / "sample_record.md"
    sample_record.write_text(
        "# sample records\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | green |\n| SAMPLE_004 | blue |\n| SAMPLE_018 | green+blue |\n",
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_018"],
        sample_record_path=sample_record,
        reference_scene_ids=["SAMPLE_001", "SAMPLE_004"],
    )

    pigment = manifest["samples"][0]["pigment_analysis"]
    assert pigment["confidence_tier"] == "review"
    assert pigment["margin"] < 0.05
    assert isinstance(pigment["review_reason"], str)
    assert pigment["review_reason"]


def test_export_workbench_manifest_marks_close_candidates_for_follow_up_review(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"

    curves = {
        "SAMPLE_001": [0.15, 0.35, 0.92, 0.55, 0.24],
        "SAMPLE_004": [0.01, 0.18, 0.68, 0.38, 0.08],
        "SAMPLE_019": [0.01, 0.16, 0.70, 0.36, 0.06],
    }
    for scene_id, curve in curves.items():
        _write_scene(scenes_root / scene_id)
        cube = np.tile(np.asarray(curve, dtype=np.float32), (2, 3, 1))
        np.save(scenes_root / scene_id / "five_band.npy", cube)
        (scenes_root / scene_id / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)).save(
            scenes_root / scene_id / "masks" / "paint.png"
        )
        _write_predictions(prediction_root, scene_id)

    sample_record = tmp_path / "sample_record.md"
    sample_record.write_text(
        "# sample records\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | green |\n| SAMPLE_004 | blue |\n| SAMPLE_019 | green+blue |\n",
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_019"],
        sample_record_path=sample_record,
        reference_scene_ids=["SAMPLE_001", "SAMPLE_004"],
    )

    pigment = manifest["samples"][0]["pigment_analysis"]
    assert 0.05 <= pigment["margin"] < 0.12
    assert pigment["confidence_tier"] == "close"
    assert pigment["verdict"] == "stable"
    assert pigment["should_review"] is True
    assert "建议复核" in pigment["message"]
    assert "领先" in pigment["review_reason"]


def test_export_workbench_manifest_marks_ambiguous_pigment_and_cluster_assets(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"

    curves = {
        "SAMPLE_001": [10.0, 20.0, 30.0, 40.0, 50.0],
        "SAMPLE_004": [11.0, 21.0, 29.0, 39.0, 49.0],
        "SAMPLE_018": [10.5, 20.5, 29.5, 39.5, 49.5],
    }
    for scene_id, curve in curves.items():
        _write_scene(scenes_root / scene_id)
        cube = np.tile(np.asarray(curve, dtype=np.float32), (2, 3, 1))
        np.save(scenes_root / scene_id / "five_band.npy", cube)
        (scenes_root / scene_id / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)).save(
            scenes_root / scene_id / "masks" / "paint.png"
        )
        _write_predictions(prediction_root, scene_id)

    sample_root = prediction_root / "SAMPLE_018"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(sample_root / "dual_pigment_overlay.png")
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(sample_root / "dual_pigment_labels.png")
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(sample_root / "dual_pigment_curves.png")
    (sample_root / "dual_pigment_summary.csv").write_text("cluster_id,label\n1,green\n", encoding="utf-8")
    (sample_root / "dual_pigment_summary.json").write_text(json.dumps({"subregions": [{"cluster_id": 1, "label": "green", "score": 0.9, "margin": 0.1}]}, ensure_ascii=False), encoding="utf-8")

    sample_record = tmp_path / "sample_record.md"
    sample_record.write_text(
        "# sample records\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | green |\n| SAMPLE_004 | blue |\n| SAMPLE_018 | green+blue |\n",
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_018"],
        sample_record_path=sample_record,
        reference_scene_ids=["SAMPLE_001", "SAMPLE_004"],
    )

    pigment = manifest["samples"][0]["pigment_analysis"]
    assert pigment["verdict"] == "ambiguous"
    assert pigment["should_review"] is True
    assert pigment["cluster_recommended"] is True
    assert pigment["cluster_analysis"]["available"] is True
    assert pigment["cluster_analysis"]["overlay"].endswith("dual_pigment_overlay.png")
    assert pigment["cluster_analysis"]["label_map"].endswith("dual_pigment_labels.png")
    assert pigment["cluster_analysis"]["summary_json"].endswith("dual_pigment_summary.json")


def test_export_workbench_manifest_includes_pigment_prediction_summary(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    _write_scene(scenes_root / "SAMPLE_017")
    _write_predictions(prediction_root, "SAMPLE_017")
    sample_root = prediction_root / "SAMPLE_017"
    (sample_root / "pigment_summary.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "predicted_label": "??+??",
                "predicted_score": 0.61,
                "top_candidates": [
                    {"name": "??+??", "score": 0.61},
                    {"name": "??", "score": 0.23},
                ],
                "paint_positive_pixels": 12,
                "paint_total_pixels": 24,
                "low_confidence": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_017"],
    )

    pigment_prediction = manifest["samples"][0]["pigment_prediction"]
    assert pigment_prediction is not None
    assert pigment_prediction["predicted_label"] == "??+??"
    assert pigment_prediction["top_candidates"][0]["name"] == "??+??"


def test_export_workbench_manifest_supports_scene_prediction_root_overrides(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    primary_prediction_root = tmp_path / "predictions_primary"
    extra_prediction_root = tmp_path / "predictions_extra"

    _write_scene(scenes_root / "SAMPLE_017")
    _write_scene(scenes_root / "SAMPLE_040")
    _write_predictions(primary_prediction_root, "SAMPLE_017")
    _write_predictions(extra_prediction_root, "SAMPLE_040")

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=primary_prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_017", "SAMPLE_040"],
        scene_prediction_roots={"SAMPLE_040": extra_prediction_root},
    )

    samples = {sample["id"]: sample for sample in manifest["samples"]}
    assert samples["SAMPLE_017"]["assets"]["combined_overlay"].startswith("../predictions_primary/")
    assert samples["SAMPLE_017"]["annotation_available"] is False
    assert samples["SAMPLE_017"]["review_assets"]["combined_overlay"].startswith("../predictions_primary/")
    assert samples["SAMPLE_040"]["assets"]["combined_overlay"].startswith("../predictions_extra/")
    assert samples["SAMPLE_040"]["annotation_available"] is False
    assert samples["SAMPLE_040"]["review_assets"]["combined_overlay"].startswith("../predictions_extra/")


def test_export_workbench_manifest_includes_version_and_provenance_metadata(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    primary_prediction_root = tmp_path / "predictions_primary"
    aging_prediction_root = tmp_path / "predictions_aging"

    _write_scene(scenes_root / "SAMPLE_041")
    _write_predictions(primary_prediction_root, "SAMPLE_041")
    _write_predictions(aging_prediction_root, "SAMPLE_041")

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=primary_prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_041"],
        version_id="final_v1",
        version_label="最终融合版 final_v1",
        version_provenance={
            "prediction_root": str(primary_prediction_root),
            "paint_root": str(primary_prediction_root),
            "pollution_root": str(primary_prediction_root),
            "aging_root": str(aging_prediction_root),
        },
    )

    assert manifest["current_version_id"] == "final_v1"
    assert manifest["versions"][0]["id"] == "final_v1"
    assert manifest["versions"][0]["label"] == "最终融合版 final_v1"
    assert manifest["versions"][0]["sample_count"] == 1
    assert manifest["versions"][0]["provenance"]["aging_root"] == str(aging_prediction_root)
    assert manifest["active_version"]["id"] == "final_v1"
    assert manifest["active_version"]["sample_count"] == 1


def test_export_workbench_manifest_supports_multiple_versions(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    primary_prediction_root = tmp_path / "predictions_primary"
    aging_prediction_root = tmp_path / "predictions_aging"

    _write_scene(scenes_root / "SAMPLE_041")
    _write_predictions(primary_prediction_root, "SAMPLE_041")
    _write_predictions(aging_prediction_root, "SAMPLE_041")

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=primary_prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_041"],
        version_id="main_v1",
        version_label="主线版 main_v1",
        additional_versions=[
            {
                "id": "aging_v1",
                "label": "aging 专项版",
                "prediction_root": str(aging_prediction_root),
                "provenance": {
                    "prediction_root": str(aging_prediction_root),
                    "aging_root": str(aging_prediction_root),
                },
            }
        ],
    )

    assert manifest["current_version_id"] == "main_v1"
    assert [item["id"] for item in manifest["versions"]] == ["main_v1", "aging_v1"]
    assert manifest["versions"][1]["provenance"]["prediction_root"] == str(aging_prediction_root)
    assert manifest["versions"][1]["samples"][0]["assets"]["combined_overlay"].startswith("../predictions_aging/")
    assert manifest["versions"][1]["samples"][0]["annotation_available"] is False
    assert manifest["versions"][1]["samples"][0]["review_assets"]["combined_overlay"].startswith("../predictions_aging/")


def test_export_workbench_manifest_preserves_non_ascii_version_labels(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    _write_scene(scenes_root / "SAMPLE_041")
    _write_predictions(prediction_root, "SAMPLE_041")

    output_path = tmp_path / "ui" / "workbench_manifest.json"
    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=output_path,
        scene_ids=["SAMPLE_041"],
        version_id="final_utf8",
        version_label="?????",
    )

    assert manifest["versions"][0]["label"] == "?????"
    assert "?????" in output_path.read_text(encoding="utf-8")
    assert "?????" in output_path.with_suffix(".js").read_text(encoding="utf-8")


def test_export_workbench_manifest_writes_three_head_sample_data(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    _write_scene(scenes_root / "CAMERA_TEST")
    _write_predictions(prediction_root, "CAMERA_TEST")

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["CAMERA_TEST"],
    )

    assert manifest["band_labels"] == [450, 550, 600, 650, 700]
    assert [head["id"] for head in manifest["heads"]] == list(HEADS)
    sample = manifest["samples"][0]
    assert sample["id"] == "CAMERA_TEST"
    assert sample["assets"]["combined_overlay"].startswith("../predictions/CAMERA_TEST/")
    assert sample["annotation_available"] is False
    assert sample["assets"]["combined_overlay"].endswith("combined_overlay.png")
    assert sample["review_assets"]["combined_overlay"].endswith("combined_overlay.png")
    assert sample["heads"]["paint"]["area_ratio"] == 2 / 6
    assert sample["heads"]["paint"]["positive_pixels"] == 2
    assert sample["heads"]["paint"]["total_pixels"] == 6
    assert sample["heads"]["pollution"]["peak_wavelength"] == 700
    assert sample["heads"]["aging"]["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert (tmp_path / "ui" / "workbench_manifest.json").exists()



def test_export_workbench_manifest_can_use_scene_pigment_root_override(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    primary_prediction_root = tmp_path / "predictions_primary"
    pigment_prediction_root = tmp_path / "predictions_pigment"

    _write_scene(scenes_root / "SAMPLE_049")
    _write_predictions(primary_prediction_root, "SAMPLE_049")
    _write_predictions(pigment_prediction_root, "SAMPLE_049")

    pigment_root = pigment_prediction_root / "SAMPLE_049"
    (pigment_root / "pigment_summary.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "predicted_label": "blue+red",
                "predicted_score": 0.91,
                "top_candidates": [{"name": "blue+red", "score": 0.91}],
                "paint_positive_pixels": 12,
                "paint_total_pixels": 24,
                "low_confidence": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    dual_root = pigment_root / "dual_pigment_analysis"
    dual_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(dual_root / "dual_pigment_overlay.png")
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(dual_root / "dual_pigment_labels.png")
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(dual_root / "dual_pigment_curves.png")
    (dual_root / "dual_pigment_summary.csv").write_text("cluster_id,label\n1,blue\n", encoding="utf-8")
    (dual_root / "dual_pigment_summary.json").write_text(
        json.dumps(
            {"subregions": [{"cluster_id": 1, "label": "blue", "score": 0.9, "margin": 0.1}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=primary_prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_049"],
        scene_pigment_roots={"SAMPLE_049": pigment_prediction_root},
    )

    sample = manifest["samples"][0]
    assert sample["assets"]["combined_overlay"].startswith("../predictions_primary/SAMPLE_049/")
    assert sample["annotation_available"] is False
    assert sample["pigment_prediction"]["predicted_label"] == "blue+red"
    cluster = sample["pigment_analysis"]["cluster_analysis"]
    assert cluster["available"] is True
    assert cluster["overlay"].startswith("../predictions_pigment/")
    assert cluster["label_map"].startswith("../predictions_pigment/")


def test_export_workbench_manifest_backfills_missing_dual_pigment_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from train import analysis_workbench as workbench

    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"

    curves = {
        "SAMPLE_001": [10.0, 20.0, 30.0, 40.0, 50.0],
        "SAMPLE_004": [50.0, 40.0, 30.0, 20.0, 10.0],
        "SAMPLE_018": [10.5, 20.5, 29.5, 39.5, 49.5],
    }
    for scene_id, curve in curves.items():
        _write_scene(scenes_root / scene_id)
        cube = np.tile(np.asarray(curve, dtype=np.float32), (2, 3, 1))
        np.save(scenes_root / scene_id / "five_band.npy", cube)
        (scenes_root / scene_id / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)).save(
            scenes_root / scene_id / "masks" / "paint.png"
        )
        _write_predictions(prediction_root, scene_id)

    sample_record = tmp_path / "sample_record.md"
    sample_record.write_text(
        "# sample records\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | green |\n| SAMPLE_004 | blue |\n| SAMPLE_018 | green+blue |\n",
        encoding="utf-8",
    )

    export_calls: list[Path] = []

    def fake_export_dual_pigment_analysis(*, scene_root: Path, output_root: Path, sample_record_path: Path, paint_mask_path: Path, reference_scenes_root: Path):
        export_calls.append(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(output_root / "dual_pigment_overlay.png")
        Image.fromarray(np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint8)).save(output_root / "dual_pigment_labels.png")
        Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(output_root / "dual_pigment_curves.png")
        (output_root / "dual_pigment_summary.csv").write_text("cluster_id,label\n1,green\n2,blue\n", encoding="utf-8")
        (output_root / "dual_pigment_summary.json").write_text(
            json.dumps({"subregions": [{"cluster_id": 1, "label": "green", "score": 0.9, "margin": 0.1}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return [output_root / "dual_pigment_overlay.png"]

    monkeypatch.setattr(workbench, "export_dual_pigment_analysis", fake_export_dual_pigment_analysis)

    manifest = workbench.export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_018"],
        sample_record_path=sample_record,
        reference_scene_ids=["SAMPLE_001", "SAMPLE_004"],
    )

    assert export_calls == [prediction_root / "SAMPLE_018" / "dual_pigment_analysis"]
    cluster = manifest["samples"][0]["pigment_analysis"]["cluster_analysis"]
    assert cluster["available"] is True
    assert cluster["label_map"].endswith("dual_pigment_labels.png")
    assert cluster["summary_json"].endswith("dual_pigment_summary.json")


def test_export_workbench_manifest_includes_annotation_audit_assets(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    _write_scene(scenes_root / "SAMPLE_055")
    _write_predictions(prediction_root, "SAMPLE_055")

    audit_root = prediction_root / "SAMPLE_055" / "annotation_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(audit_root / "paint_missing_overlay.png")
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(audit_root / "paint_missing_mask.png")
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(audit_root / "paint_overmark_overlay.png")
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(audit_root / "paint_overmark_mask.png")
    (audit_root / "audit_summary.json").write_text(
        json.dumps(
            {
                "available": True,
                "target": "paint",
                "heads": {
                    "paint": {
                        "missing": {
                            "positive_pixels": 42,
                            "component_count": 2,
                            "overlay": "paint_missing_overlay.png",
                            "mask": "paint_missing_mask.png",
                        },
                        "overmark": {
                            "positive_pixels": 7,
                            "component_count": 1,
                            "overlay": "paint_overmark_overlay.png",
                            "mask": "paint_overmark_mask.png",
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_055"],
    )

    audit = manifest["samples"][0]["annotation_audit"]
    assert audit["available"] is True
    assert audit["target"] == "paint"
    assert audit["heads"]["paint"]["missing"]["positive_pixels"] == 42
    assert audit["heads"]["paint"]["missing"]["overlay"].endswith("paint_missing_overlay.png")
    assert audit["heads"]["paint"]["overmark"]["mask"].endswith("paint_overmark_mask.png")


def _write_annotation_masks(scene_root: Path) -> None:
    masks_root = scene_root / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)
    masks = {
        "paint": np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8),
        "pollution": np.array([[0, 255, 0], [0, 255, 0]], dtype=np.uint8),
        "aging": np.array([[0, 0, 255], [0, 0, 255]], dtype=np.uint8),
    }
    for head_name, mask in masks.items():
        Image.fromarray(mask).save(masks_root / f"{head_name}.png")


def test_export_workbench_manifest_prefers_annotation_assets_and_keeps_review_assets(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    scene_root = scenes_root / "SAMPLE_055"
    _write_scene(scene_root)
    _write_annotation_masks(scene_root)
    _write_predictions(prediction_root, "SAMPLE_055")

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_055"],
    )

    sample = manifest["samples"][0]
    assert sample["assets"]["combined_overlay"].startswith("generated_overlays/SAMPLE_055/")
    assert sample["assets"]["combined_overlay"].endswith("annotation_overlay.png")
    assert sample["annotation_available"] is True
    assert sample["assets"]["review_combined_overlay"].endswith("predictions/SAMPLE_055/combined_overlay.png")
    assert sample["heads"]["paint"]["mask"].endswith("SAMPLE_055/masks/paint.png")
    assert sample["heads"]["paint"]["overlay"].startswith("generated_overlays/SAMPLE_055/")
    assert sample["heads"]["paint"]["overlay"].endswith("annotation_paint_overlay.png")
    assert sample["review_heads"]["paint"]["mask"].endswith("predictions/SAMPLE_055/paint_pred.png")
    assert sample["review_heads"]["paint"]["overlay"].endswith("predictions/SAMPLE_055/paint_overlay.png")


def test_export_workbench_manifest_generates_annotation_overlays_in_cache_directory(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    scene_root = scenes_root / "SAMPLE_055"
    _write_scene(scene_root)
    _write_annotation_masks(scene_root)
    _write_predictions(prediction_root, "SAMPLE_055")

    export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_055"],
    )

    generated_root = tmp_path / "ui" / "generated_overlays" / "SAMPLE_055"
    assert (generated_root / "annotation_overlay.png").exists()
    assert (generated_root / "annotation_paint_overlay.png").exists()
    assert (generated_root / "annotation_pollution_overlay.png").exists()
    assert (generated_root / "annotation_aging_overlay.png").exists()
    assert not (scene_root / "annotation_overlay.png").exists()
    assert not (scene_root / "annotation_paint_overlay.png").exists()


def test_export_workbench_manifest_keeps_existing_annotation_overlays_when_replace_is_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    scene_root = scenes_root / "SAMPLE_055"
    _write_scene(scene_root)
    _write_annotation_masks(scene_root)
    _write_predictions(prediction_root, "SAMPLE_055")
    output_path = tmp_path / "ui" / "workbench_manifest.json"

    export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=output_path,
        scene_ids=["SAMPLE_055"],
    )

    generated_root = tmp_path / "ui" / "generated_overlays" / "SAMPLE_055"
    combined_path = generated_root / "annotation_overlay.png"
    assert combined_path.exists()

    paint_mask_path = scene_root / "masks" / "paint.png"
    updated_time = combined_path.stat().st_mtime + 5
    import os
    os.utime(paint_mask_path, (updated_time, updated_time))

    from train import analysis_workbench as module

    real_replace = module.os.replace
    hit = {"locked": False}

    def locked_replace(src, dst):
        if Path(dst) == combined_path and not hit["locked"]:
            hit["locked"] = True
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(module.os, "replace", locked_replace)

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=output_path,
        scene_ids=["SAMPLE_055"],
    )

    assert hit["locked"] is True
    assert manifest["samples"][0]["assets"]["combined_overlay"].endswith("annotation_overlay.png")
    assert combined_path.exists()


def test_export_workbench_manifest_reuses_cached_annotation_overlays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    scene_root = scenes_root / "SAMPLE_055"
    _write_scene(scene_root)
    _write_annotation_masks(scene_root)
    _write_predictions(prediction_root, "SAMPLE_055")
    output_path = tmp_path / "ui" / "workbench_manifest.json"

    export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=output_path,
        scene_ids=["SAMPLE_055"],
    )

    original_save = Image.Image.save
    save_calls: list[str] = []

    def tracking_save(self, fp, *args, **kwargs):
        save_calls.append(str(fp))
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", tracking_save)

    export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=output_path,
        scene_ids=["SAMPLE_055"],
    )

    overlay_calls = [item for item in save_calls if "generated_overlays" in item and item.endswith('.png')]
    assert overlay_calls == []


def test_export_workbench_manifest_combined_annotation_overlay_keeps_pollution_and_aging_visible(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    prediction_root = tmp_path / "predictions"
    scene_root = scenes_root / "SAMPLE_055"
    _write_scene(scene_root)
    _write_annotation_masks(scene_root)
    _write_predictions(prediction_root, "SAMPLE_055")

    manifest = export_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=tmp_path / "ui" / "workbench_manifest.json",
        scene_ids=["SAMPLE_055"],
    )

    sample = manifest["samples"][0]
    overlay_path = (tmp_path / "ui" / sample["assets"]["combined_overlay"]).resolve()
    combined = np.asarray(Image.open(overlay_path).convert("RGB"), dtype=np.uint8)

    paint_pixel = combined[0, 0]
    pollution_pixel = combined[0, 1]
    aging_pixel = combined[0, 2]

    assert paint_pixel[0] > paint_pixel[1]
    assert pollution_pixel[1] >= pollution_pixel[2]
    assert pollution_pixel[0] > 40
    assert aging_pixel[2] > aging_pixel[0]

