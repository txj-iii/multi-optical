from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.export_curves import BAND_LABELS, compute_band_means, configure_matplotlib_fonts, load_sample_pigments
from train.mixed_pigment_subregions import (
    analyze_mixed_paint_region,
    cluster_paint_region_spectra,
    normalize_vector,
)

CLUSTER_COLORS = {
    1: np.array([220, 60, 60], dtype=np.float32),
    2: np.array([40, 120, 230], dtype=np.float32),
}


def build_pigment_prototypes(scenes_root: Path, sample_record_path: Path) -> dict[str, np.ndarray]:
    pigments = load_sample_pigments(sample_record_path)
    prototype_values: dict[str, list[list[float]]] = {}
    for scene_root in sorted(path for path in scenes_root.iterdir() if path.is_dir() and (path / "five_band.npy").exists()):
        sample_id = scene_root.name
        pigment = pigments.get(sample_id)
        if pigment is None or pigment == "无颜料" or "+" in pigment:
            continue
        mask_path = scene_root / "masks" / "paint.png"
        if not mask_path.exists():
            continue
        paint_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if not np.any(paint_mask > 0):
            continue
        five_band = np.load(scene_root / "five_band.npy").astype(np.float32)
        prototype_values.setdefault(pigment, []).append(compute_band_means(five_band, paint_mask))

    prototypes: dict[str, np.ndarray] = {}
    for pigment, values in prototype_values.items():
        prototypes[pigment] = normalize_vector(np.mean(np.asarray(values, dtype=np.float32), axis=0))
    return prototypes


def assign_cluster_pigments(centers: np.ndarray, prototypes: dict[str, np.ndarray]) -> list[str]:
    if not prototypes:
        return [f"类别{index + 1}" for index in range(len(centers))]

    available = dict(prototypes)
    labels: list[str] = []
    for center in centers:
        normalized_center = normalize_vector(center)
        best_label = min(
            available.items(),
            key=lambda item: float(np.linalg.norm(normalized_center - item[1])),
        )[0]
        labels.append(best_label)
        if len(available) > 1:
            available.pop(best_label, None)
    return labels


def build_cluster_overlay(preview: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    overlay = preview.copy().astype(np.float32)
    for cluster_index, color in CLUSTER_COLORS.items():
        positive = label_map == cluster_index
        if np.any(positive):
            overlay[positive] = overlay[positive] * 0.4 + color * 0.6
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_cluster_curves(centers: np.ndarray, labels: Sequence[str], output_path: Path) -> None:
    x_values = [int(label) for label in BAND_LABELS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, center in enumerate(centers, start=1):
        ax.plot(
            x_values,
            center.tolist(),
            marker="o",
            linewidth=2,
            color=(CLUSTER_COLORS[index] / 255.0).tolist(),
            label=labels[index - 1],
        )
    ax.set_title("Mixed pigment subregion mean curves")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean intensity")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_dual_pigment_analysis(
    *,
    scene_root: Path,
    output_root: Path,
    sample_record_path: Path,
    paint_mask_path: Path,
    reference_scenes_root: Path,
) -> list[Path]:
    configure_matplotlib_fonts()
    output_root.mkdir(parents=True, exist_ok=True)

    five_band = np.load(scene_root / "five_band.npy").astype(np.float32)
    preview = np.asarray(Image.open(scene_root / "preview.png").convert("RGB"))
    paint_mask = np.asarray(Image.open(paint_mask_path).convert("L"), dtype=np.uint8)
    label_map, centers = cluster_paint_region_spectra(five_band, paint_mask)
    prototypes = build_pigment_prototypes(reference_scenes_root, sample_record_path)
    cluster_labels = assign_cluster_pigments(centers, prototypes)
    analysis = analyze_mixed_paint_region(
        five_band=five_band,
        paint_mask=paint_mask,
        prototypes={
            name: {"normalized": value.tolist(), "values": value.tolist()}
            for name, value in prototypes.items()
        },
        sample_id=scene_root.name,
    )

    overlay = build_cluster_overlay(preview, label_map)
    overlay_path = output_root / "dual_pigment_overlay.png"
    mask_path = output_root / "dual_pigment_labels.png"
    csv_path = output_root / "dual_pigment_summary.csv"
    json_path = output_root / "dual_pigment_summary.json"
    curve_path = output_root / "dual_pigment_curves.png"
    log_path = output_root / "dual_pigment_log.txt"

    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(label_map).save(mask_path)
    plot_cluster_curves(centers, cluster_labels, curve_path)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["cluster_id", "label", *(f"band_{label}" for label in BAND_LABELS)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_index, (cluster_label, center) in enumerate(zip(cluster_labels, centers), start=1):
            row = {"cluster_id": cluster_index, "label": cluster_label}
            for band_label, value in zip(BAND_LABELS, center.tolist()):
                row[f"band_{band_label}"] = float(value)
            writer.writerow(row)

    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                f"scene_id={scene_root.name}",
                f"paint_mask_path={paint_mask_path}",
                f"cluster_count={len(centers)}",
                *[f"cluster_{index + 1}_label={label}" for index, label in enumerate(cluster_labels)],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [overlay_path, mask_path, csv_path, json_path, curve_path, log_path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a two-cluster pigment analysis inside a predicted paint region.")
    parser.add_argument("--scene-root", type=str, required=True, help="Scene directory that contains five_band.npy and preview.png.")
    parser.add_argument("--paint-mask-path", type=str, required=True, help="Paint mask path, usually paint_pred.png.")
    parser.add_argument("--output-root", type=str, required=True, help="Output directory for overlay, curves, and csv.")
    parser.add_argument(
        "--sample-record-path",
        type=str,
        default=str(PROJECT_ROOT / "readme" / "样本记录规范.md"),
        help="Markdown file containing sample-to-pigment mapping.",
    )
    parser.add_argument(
        "--reference-scenes-root",
        type=str,
        default=str(PROJECT_ROOT / "train" / "camera_eval_workspace"),
        help="Scenes root used to build single-pigment spectral prototypes.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    exported = export_dual_pigment_analysis(
        scene_root=Path(args.scene_root),
        output_root=Path(args.output_root),
        sample_record_path=Path(args.sample_record_path),
        paint_mask_path=Path(args.paint_mask_path),
        reference_scenes_root=Path(args.reference_scenes_root),
    )
    print(f"exported={len(exported)}")
    print(f"output_root={Path(args.output_root)}")


if __name__ == "__main__":
    main()
