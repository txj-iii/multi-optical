from pathlib import Path

import numpy as np
from PIL import Image

from train.evaluate_key_scenes import (
    compute_binary_metrics,
    compare_summary_against_baseline,
    evaluate_prediction_root,
    summarize_metrics,
)


def test_compute_binary_metrics_reports_zero_gt_false_positive_area() -> None:
    gt = np.zeros((2, 2), dtype=bool)
    pred = np.array([[True, False], [True, False]])

    metrics = compute_binary_metrics(pred, gt)

    assert metrics["gt_pixels"] == 0
    assert metrics["pred_pixels"] == 2
    assert metrics["zero_gt_fp_area"] == 2
    assert metrics["precision"] == 0.0


def test_summarize_metrics_tracks_focus_and_global_behavior() -> None:
    rows = [
        {
            "scene": "SAMPLE_016",
            "head": "paint",
            "gt_pixels": 10,
            "pred_pixels": 8,
            "precision": 0.75,
            "recall": 0.6,
            "iou": 0.5,
            "zero_gt_fp_area": 0,
        },
        {
            "scene": "SAMPLE_020",
            "head": "paint",
            "gt_pixels": 0,
            "pred_pixels": 3,
            "precision": 0.0,
            "recall": 1.0,
            "iou": 0.0,
            "zero_gt_fp_area": 3,
        },
        {
            "scene": "SAMPLE_003",
            "head": "paint",
            "gt_pixels": 6,
            "pred_pixels": 6,
            "precision": 0.5,
            "recall": 0.5,
            "iou": 1 / 3,
            "zero_gt_fp_area": 0,
        },
    ]

    summary = summarize_metrics(rows)

    assert summary["paint"]["positive_scene_count"] == 2
    assert summary["paint"]["focus_positive_scene_count"] == 1
    assert summary["paint"]["global_recall_avg"] == 0.55
    assert summary["paint"]["focus_recall_avg"] == 0.6
    assert summary["paint"]["zero_gt_fp_area_avg"] == 3.0


def test_compare_summary_against_baseline_flags_regression_outside_focus() -> None:
    baseline = {
        "paint": {
            "global_recall_avg": 0.62,
            "global_precision_avg": 0.48,
            "focus_recall_avg": 0.4,
            "focus_precision_avg": 0.3,
            "zero_gt_fp_area_avg": 900.0,
        }
    }
    candidate = {
        "paint": {
            "global_recall_avg": 0.51,
            "global_precision_avg": 0.47,
            "focus_recall_avg": 0.52,
            "focus_precision_avg": 0.33,
            "zero_gt_fp_area_avg": 200.0,
        }
    }

    regressions = compare_summary_against_baseline(
        candidate,
        baseline,
        recall_tolerance=0.05,
        precision_tolerance=0.02,
        zero_gt_fp_tolerance=0.0,
    )

    assert regressions == [
        "paint global_recall_avg regressed: baseline=0.6200 candidate=0.5100 tolerance=0.0500"
    ]


def test_evaluate_prediction_root_reads_masks_and_summarizes(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    scenes_root = tmp_path / "scenes"
    scene_id = "SAMPLE_016"
    (prediction_root / scene_id).mkdir(parents=True)
    (scenes_root / scene_id / "masks").mkdir(parents=True)

    paint_pred = np.zeros((4, 4), dtype=np.uint8)
    paint_pred[1:3, 1:3] = 255
    paint_gt = np.zeros((4, 4), dtype=np.uint8)
    paint_gt[2:, 2:] = 255
    empty = np.zeros((4, 4), dtype=np.uint8)

    Image.fromarray(paint_pred).save(prediction_root / scene_id / "paint_pred.png")
    Image.fromarray(empty).save(prediction_root / scene_id / "pollution_pred.png")
    Image.fromarray(empty).save(prediction_root / scene_id / "aging_pred.png")
    Image.fromarray(paint_gt).save(scenes_root / scene_id / "masks" / "paint.png")
    Image.fromarray(empty).save(scenes_root / scene_id / "masks" / "pollution.png")
    Image.fromarray(empty).save(scenes_root / scene_id / "masks" / "aging.png")

    rows, summary = evaluate_prediction_root(
        prediction_root=prediction_root,
        scenes_root=scenes_root,
        scene_ids=[scene_id],
    )

    assert len(rows) == 3
    assert summary["paint"]["focus_positive_scene_count"] == 1
    assert summary["paint"]["focus_recall_avg"] == 0.25
