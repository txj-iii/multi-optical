"""Evaluate background4_v1 predictions against manually saved scene masks."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "train" / "camera_eval_workspace"
STATE = ROOT / "ui" / "analysis_workbench" / "workflow_state.json"
PREDICTIONS = ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "background4_v1"
HEADS = ("paint", "pollution", "aging")


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float | None:
    pred = prediction > 0
    truth = target > 0
    union = np.logical_or(pred, truth).sum()
    return None if union == 0 else float(np.logical_and(pred, truth).sum() / union)


def key_for(record: dict[str, object]) -> str:
    return f"背景={record.get('background_role') or '未知'} | 光照={record.get('light_level') or '未知'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense background4_v1 segmentation and pigment predictions")
    parser.add_argument("--prediction-root", type=Path, default=PREDICTIONS)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"samples": {}}
    records = state.get("samples", {})
    scene_ids = args.scene_id or sorted(path.name for path in args.prediction_root.glob("SAMPLE_*") if path.is_dir())
    grouped: dict[str, dict[str, list[float] | Counter[tuple[int, int]] | int]] = defaultdict(lambda: {**{head: [] for head in HEADS}, "background_paint_false_positive_pixels": 0, "background_pixels": 0, "pigment_confusion": Counter(), "pigment_valid_pixels": 0, "pigment_ignore_pixels": 0})
    evaluated: list[str] = []
    for scene_id in scene_ids:
        scene_root, prediction_root = SCENES / scene_id, args.prediction_root / scene_id
        if not scene_root.exists() or not prediction_root.exists():
            continue
        record = records.get(scene_id, {})
        group = grouped[key_for(record)]
        for head in HEADS:
            target_path, pred_path = scene_root / "masks" / f"{head}.png", prediction_root / f"{head}_pred.png"
            if not target_path.exists() or not pred_path.exists():
                continue
            score = binary_iou(read_image(pred_path), read_image(target_path))
            if score is not None:
                group[head].append(score)  # type: ignore[index]
        paint_path, paint_pred_path = scene_root / "masks" / "paint.png", prediction_root / "paint_pred.png"
        if paint_path.exists() and paint_pred_path.exists():
            paint, predicted_paint = read_image(paint_path), read_image(paint_pred_path)
            if not (paint > 0).any():
                group["background_paint_false_positive_pixels"] += int((predicted_paint > 0).sum())  # type: ignore[operator]
                group["background_pixels"] += int(paint.size)  # type: ignore[operator]
        pigment_path, pigment_pred_path = scene_root / "masks" / "pigment.png", prediction_root / "pigment_pred.png"
        if pigment_path.exists() and pigment_pred_path.exists() and paint_path.exists():
            pigment, predicted, paint = read_image(pigment_path), read_image(pigment_pred_path), read_image(paint_path)
            valid = (paint > 0) & (pigment > 0)
            group["pigment_valid_pixels"] += int(valid.sum())  # type: ignore[operator]
            group["pigment_ignore_pixels"] += int(((paint > 0) & (pigment == 0)).sum())  # type: ignore[operator]
            for truth, guess in zip(pigment[valid].tolist(), predicted[valid].tolist()):
                group["pigment_confusion"][(int(truth), int(guess))] += 1  # type: ignore[index]
        evaluated.append(scene_id)
    report_groups: dict[str, object] = {}
    for name, values in grouped.items():
        valid = int(values["pigment_valid_pixels"])
        confusion = values["pigment_confusion"]
        correct = sum(count for (truth, guess), count in confusion.items() if truth == guess)
        report_groups[name] = {
            "head_iou": {head: (float(np.mean(values[head])) if values[head] else None) for head in HEADS},
            "background_paint_false_positive_ratio": (values["background_paint_false_positive_pixels"] / values["background_pixels"] if values["background_pixels"] else None),
            "pigment_accuracy_in_manual_paint": (correct / valid if valid else None),
            "pigment_ignore_ratio_in_manual_paint": (values["pigment_ignore_pixels"] / (valid + values["pigment_ignore_pixels"]) if valid + values["pigment_ignore_pixels"] else None),
            "pigment_confusion": {f"{truth}->{guess}": count for (truth, guess), count in sorted(confusion.items())},
        }
    report = {"version_id": "background4_v1", "evaluated_scenes": evaluated, "groups": report_groups}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
