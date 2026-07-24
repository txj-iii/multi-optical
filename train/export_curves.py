from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.predict import _resolve_scene_roots

BAND_LABELS = ("450", "550", "600", "650", "700")
PIGMENT_COLOR_FAMILIES: dict[str, list[tuple[float, float, float]]] = {
    "石绿": [(0.18, 0.55, 0.34), (0.28, 0.68, 0.42), (0.44, 0.78, 0.55)],
    "石青": [(0.16, 0.39, 0.76), (0.28, 0.50, 0.86), (0.42, 0.63, 0.92)],
    "朱砂": [(0.74, 0.20, 0.18), (0.84, 0.31, 0.26), (0.91, 0.45, 0.40)],
    "代赭": [(0.55, 0.33, 0.18), (0.67, 0.42, 0.24), (0.79, 0.53, 0.34)],
}
DEFAULT_COLOR_FAMILY = [(0.40, 0.40, 0.40), (0.55, 0.55, 0.55), (0.70, 0.70, 0.70)]


def configure_matplotlib_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available_fonts:
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams.get("font.sans-serif", [])]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_sample_pigments(sample_record_path: Path) -> dict[str, str]:
    pigments: dict[str, str] = {}
    lines = sample_record_path.read_text(encoding="utf-8").splitlines()
    in_target_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| sample_id ") and "| pigment |" in stripped:
            in_target_table = True
            continue
        if in_target_table and stripped.startswith("## "):
            break
        if not in_target_table or not stripped.startswith("| SAMPLE_"):
            continue
        columns = [part.strip() for part in stripped.strip("|").split("|")]
        if len(columns) < 2:
            continue
        pigments[columns[0]] = columns[1]
    return pigments


def compute_band_means(five_band: np.ndarray, paint_mask: np.ndarray) -> list[float]:
    positive = paint_mask > 0
    if not np.any(positive):
        raise ValueError("paint mask is empty")
    return [float(five_band[:, :, channel_index][positive].mean()) for channel_index in range(five_band.shape[2])]


def normalize_curve_values(values: Sequence[float]) -> list[float]:
    max_value = max(float(value) for value in values)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [float(value) / max_value for value in values]


def resolve_curve_style(sample_id: str, pigment: str) -> tuple[tuple[float, float, float], str]:
    family = PIGMENT_COLOR_FAMILIES.get(pigment, DEFAULT_COLOR_FAMILY)
    sample_number = int(sample_id.split("_")[-1])
    return family[(sample_number - 1) % len(family)], pigment


def _plot_single_curve(sample_id: str, pigment: str, values: Sequence[float], output_path: Path) -> None:
    x_values = [int(label) for label in BAND_LABELS]
    color, _ = resolve_curve_style(sample_id, pigment)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_values, values, marker="o", linewidth=2, color=color)
    ax.set_title(f"{sample_id} - {pigment}")
    ax.set_xlabel("波段 (nm)")
    ax.set_ylabel("paint区域平均强度")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_combined_curves(rows: Sequence[dict[str, object]], output_path: Path, *, normalized: bool) -> None:
    x_values = [int(label) for label in BAND_LABELS]
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        values = [float(row[f"band_{label}"]) for label in BAND_LABELS]
        if normalized:
            values = normalize_curve_values(values)
        color, legend_group = resolve_curve_style(str(row["sample_id"]), str(row["pigment"]))
        ax.plot(
            x_values,
            values,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{row['sample_id']} {legend_group}",
        )
    ax.set_title("5波段 paint区域曲线" if not normalized else "5波段 paint区域归一化曲线")
    ax.set_xlabel("波段 (nm)")
    ax.set_ylabel("paint区域平均强度" if not normalized else "归一化强度")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_sample_curves(
    scenes_root: Path,
    output_root: Path,
    sample_record_path: Path,
    scene_ids: Sequence[str] | None = None,
    use_prediction_mask: bool = False,
    prediction_root: Path | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    configure_matplotlib_fonts()
    pigments = load_sample_pigments(sample_record_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    scene_roots = _resolve_scene_roots(scenes_root, scene_ids)
    for scene_root in scene_roots:
        sample_id = scene_root.name
        if sample_id not in pigments:
            continue
        five_band = np.load(scene_root / "five_band.npy").astype(np.float32)
        if five_band.shape[2] != len(BAND_LABELS):
            raise ValueError(f"{sample_id} does not contain {len(BAND_LABELS)} channels.")

        if use_prediction_mask:
            if prediction_root is None:
                raise ValueError("prediction_root is required when use_prediction_mask=True")
            mask_path = prediction_root / sample_id / "paint_pred.png"
        else:
            mask_path = scene_root / "masks" / "paint.png"

        paint_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        try:
            means = compute_band_means(five_band, paint_mask)
        except ValueError:
            skipped.append(sample_id)
            continue

        row: dict[str, object] = {
            "sample_id": sample_id,
            "pigment": pigments[sample_id],
        }
        for band_label, value in zip(BAND_LABELS, means):
            row[f"band_{band_label}"] = value
        rows.append(row)
        _plot_single_curve(sample_id, pigments[sample_id], means, output_root / f"{sample_id}_curve.png")

    csv_path = output_root / "curve_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "pigment", *(f"band_{label}" for label in BAND_LABELS)],
        )
        writer.writeheader()
        writer.writerows(rows)

    log_lines = [
        f"use_prediction_mask={use_prediction_mask}",
        f"scene_count={len(scene_roots)}",
        f"exported_count={len(rows)}",
        f"skipped_count={len(skipped)}",
    ]
    for sample_id in skipped:
        log_lines.append(f"skipped={sample_id}: empty paint mask")
    (output_root / "curve_export_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if rows:
        _plot_combined_curves(rows, output_root / "all_samples_curves.png", normalized=False)
        _plot_combined_curves(rows, output_root / "all_samples_curves_normalized.png", normalized=True)
    return rows, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export five-band paint-region mean curves.")
    parser.add_argument(
        "--scenes-root",
        type=str,
        default=str(PROJECT_ROOT / "train" / "camera_eval_workspace"),
        help="Directory containing scene folders with five_band.npy and masks.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "train" / "experiments" / "five_band_curves" / "task_specific" / "epochs_3"),
        help="Output directory for curve csv and png files.",
    )
    parser.add_argument(
        "--sample-record-path",
        type=str,
        default=str(PROJECT_ROOT / "readme" / "样本记录规范.md"),
        help="Markdown file containing sample-to-pigment mapping.",
    )
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        default=None,
        help="Optional scene IDs to export, for example SAMPLE_001 SAMPLE_002.",
    )
    parser.add_argument(
        "--use-prediction-mask",
        action="store_true",
        help="Read paint_pred.png from the prediction output root instead of the ground-truth paint mask.",
    )
    parser.add_argument(
        "--prediction-root",
        type=str,
        default=None,
        help="Prediction output root that contains <scene_id>/paint_pred.png.",
    )
    return parser.parse_args(argv)


def main() -> None:
    configure_matplotlib_fonts()
    args = parse_args()
    rows, skipped = export_sample_curves(
        scenes_root=Path(args.scenes_root),
        output_root=Path(args.output_root),
        sample_record_path=Path(args.sample_record_path),
        scene_ids=args.scene_ids,
        use_prediction_mask=args.use_prediction_mask,
        prediction_root=Path(args.prediction_root) if args.prediction_root else None,
    )
    print(f"exported={len(rows)}")
    print(f"skipped={len(skipped)}")
    print(f"output_root={Path(args.output_root)}")


if __name__ == "__main__":
    main()
