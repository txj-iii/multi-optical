from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

HEAD_NAMES = ("paint", "pollution", "aging")
FOCUS_SCENE_IDS = (
    "SAMPLE_015",
    "SAMPLE_016",
    "SAMPLE_017",
    "SAMPLE_019",
    "SAMPLE_020",
    "SAMPLE_021",
    "SAMPLE_025",
    "SAMPLE_026",
    "SAMPLE_027",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions on key and global scenes.")
    parser.add_argument("--prediction-root", type=str, required=True, help="Prediction root with per-scene outputs.")
    parser.add_argument("--scenes-root", type=str, required=True, help="Ground-truth scene root with masks.")
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Optional scene IDs to evaluate.")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional metrics CSV output path.")
    parser.add_argument("--summary-output", type=str, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--baseline-summary", type=str, default=None, help="Optional baseline summary JSON path.")
    parser.add_argument("--recall-tolerance", type=float, default=0.02, help="Allowed global recall drop.")
    parser.add_argument("--precision-tolerance", type=float, default=0.02, help="Allowed global precision drop.")
    parser.add_argument(
        "--zero-gt-fp-tolerance",
        type=float,
        default=0.0,
        help="Allowed increase in zero-GT false-positive area average.",
    )
    return parser.parse_args(argv)


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    gt_pixels = int(gt_bool.sum())
    pred_pixels = int(pred_bool.sum())
    if tp + fp:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if gt_pixels == 0 and pred_pixels == 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return {
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "zero_gt_fp_area": pred_pixels if gt_pixels == 0 else 0,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def summarize_metrics(rows: Sequence[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    focus_scene_ids = set(FOCUS_SCENE_IDS)
    for head_name in HEAD_NAMES:
        head_rows = [row for row in rows if row["head"] == head_name]
        positive_rows = [row for row in head_rows if int(row["gt_pixels"]) > 0]
        focus_rows = [row for row in head_rows if str(row["scene"]) in focus_scene_ids]
        focus_positive_rows = [row for row in focus_rows if int(row["gt_pixels"]) > 0]
        zero_gt_rows = [row for row in head_rows if int(row["gt_pixels"]) == 0]
        summary[head_name] = {
            "scene_count": len(head_rows),
            "focus_scene_count": len(focus_rows),
            "positive_scene_count": len(positive_rows),
            "focus_positive_scene_count": len(focus_positive_rows),
            "global_recall_avg": _mean(float(row["recall"]) for row in positive_rows),
            "global_precision_avg": _mean(float(row["precision"]) for row in positive_rows),
            "focus_recall_avg": _mean(float(row["recall"]) for row in focus_positive_rows),
            "focus_precision_avg": _mean(float(row["precision"]) for row in focus_positive_rows),
            "global_iou_avg": _mean(float(row["iou"]) for row in positive_rows),
            "focus_iou_avg": _mean(float(row["iou"]) for row in focus_positive_rows),
            "zero_gt_fp_area_avg": _mean(float(row["zero_gt_fp_area"]) for row in zero_gt_rows),
        }
    return summary


def compare_summary_against_baseline(
    candidate: dict[str, dict[str, float | int]],
    baseline: dict[str, dict[str, float | int]],
    *,
    recall_tolerance: float,
    precision_tolerance: float,
    zero_gt_fp_tolerance: float,
) -> list[str]:
    regressions: list[str] = []
    for head_name in HEAD_NAMES:
        candidate_head = candidate.get(head_name)
        baseline_head = baseline.get(head_name)
        if not candidate_head or not baseline_head:
            continue
        comparisons = (
            ("global_recall_avg", recall_tolerance, "lower"),
            ("global_precision_avg", precision_tolerance, "lower"),
            ("zero_gt_fp_area_avg", zero_gt_fp_tolerance, "higher"),
        )
        for metric_name, tolerance, direction in comparisons:
            baseline_value = float(baseline_head.get(metric_name, 0.0))
            candidate_value = float(candidate_head.get(metric_name, 0.0))
            if direction == "lower" and candidate_value < baseline_value - tolerance:
                regressions.append(
                    f"{head_name} {metric_name} regressed: baseline={baseline_value:.4f} "
                    f"candidate={candidate_value:.4f} tolerance={tolerance:.4f}"
                )
            if direction == "higher" and candidate_value > baseline_value + tolerance:
                regressions.append(
                    f"{head_name} {metric_name} regressed: baseline={baseline_value:.4f} "
                    f"candidate={candidate_value:.4f} tolerance={tolerance:.4f}"
                )
    return regressions


def evaluate_prediction_root(
    *,
    prediction_root: Path,
    scenes_root: Path,
    scene_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, float | int]]]:
    resolved_scene_ids = list(scene_ids) if scene_ids else sorted(
        path.name for path in prediction_root.iterdir() if path.is_dir()
    )
    rows: list[dict[str, float | int | str]] = []
    for scene_id in resolved_scene_ids:
        for head_name in HEAD_NAMES:
            pred = np.asarray(Image.open(prediction_root / scene_id / f"{head_name}_pred.png").convert("L")) > 0
            gt = np.asarray(Image.open(scenes_root / scene_id / "masks" / f"{head_name}.png").convert("L")) > 0
            rows.append(
                {
                    "scene": scene_id,
                    "head": head_name,
                    **compute_binary_metrics(pred, gt),
                }
            )
    return rows, summarize_metrics(rows)


def _write_rows_csv(rows: Sequence[dict[str, float | int | str]], csv_path: Path) -> None:
    fieldnames = ["scene", "head", "gt_pixels", "pred_pixels", "precision", "recall", "iou", "zero_gt_fp_area"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(summary: dict[str, dict[str, float | int]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, summary = evaluate_prediction_root(
        prediction_root=Path(args.prediction_root),
        scenes_root=Path(args.scenes_root),
        scene_ids=args.scene_ids,
    )
    if args.csv_output:
        _write_rows_csv(rows, Path(args.csv_output))
    if args.summary_output:
        _write_summary_json(summary, Path(args.summary_output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.baseline_summary:
        baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
        regressions = compare_summary_against_baseline(
            summary,
            baseline,
            recall_tolerance=args.recall_tolerance,
            precision_tolerance=args.precision_tolerance,
            zero_gt_fp_tolerance=args.zero_gt_fp_tolerance,
        )
        if regressions:
            raise SystemExit("\n".join(regressions))


if __name__ == "__main__":
    main()
