from pathlib import Path

import numpy as np
from PIL import Image

from train.export_dual_pigment_analysis import (
    assign_cluster_pigments,
    build_pigment_prototypes,
    cluster_paint_region_spectra,
    export_dual_pigment_analysis,
    normalize_vector,
)


def test_cluster_paint_region_spectra_separates_two_spectral_groups() -> None:
    five_band = np.zeros((2, 4, 5), dtype=np.float32)
    five_band[:, :2, :] = np.array([1.0, 1.2, 1.4, 1.6, 1.8], dtype=np.float32)
    five_band[:, 2:, :] = np.array([4.0, 3.8, 3.6, 3.4, 3.2], dtype=np.float32)
    paint_mask = np.full((2, 4), 255, dtype=np.uint8)

    label_map, centers = cluster_paint_region_spectra(five_band, paint_mask, cluster_count=2, seed=0)

    assert set(np.unique(label_map).tolist()) == {1, 2}
    assert centers.shape == (2, 5)
    left_label = int(label_map[0, 0])
    right_label = int(label_map[0, 3])
    assert left_label != right_label


def test_assign_cluster_pigments_uses_nearest_normalized_prototype() -> None:
    prototypes = {
        "石青": normalize_vector([1.0, 1.2, 1.4, 1.6, 1.8]),
        "朱砂": normalize_vector([4.0, 3.8, 3.6, 3.4, 3.2]),
    }
    centers = np.asarray(
        [
            [1.0, 1.1, 1.4, 1.6, 1.7],
            [4.1, 3.7, 3.5, 3.3, 3.1],
        ],
        dtype=np.float32,
    )

    labels = assign_cluster_pigments(centers, prototypes)

    assert labels == ["石青", "朱砂"]


def test_export_dual_pigment_analysis_writes_overlay_curves_and_csv(tmp_path: Path) -> None:
    sample_record = tmp_path / "samples.md"
    sample_record.write_text(
        "# 样本记录规范\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | 石青 |\n| SAMPLE_002 | 朱砂 |\n| SAMPLE_017 | 石青+朱砂 |\n",
        encoding="utf-8",
    )
    reference_root = tmp_path / "reference"
    for scene_id, curve in {
        "SAMPLE_001": [1.0, 1.2, 1.4, 1.6, 1.8],
        "SAMPLE_002": [4.0, 3.8, 3.6, 3.4, 3.2],
    }.items():
        scene_root = reference_root / scene_id
        scene_root.mkdir(parents=True)
        cube = np.tile(np.asarray(curve, dtype=np.float32), (4, 4, 1))
        np.save(scene_root / "five_band.npy", cube)
        masks_root = scene_root / "masks"
        masks_root.mkdir()
        Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(masks_root / "paint.png")

    scene_root = tmp_path / "scene" / "SAMPLE_017"
    scene_root.mkdir(parents=True)
    cube = np.zeros((4, 4, 5), dtype=np.float32)
    cube[:, :2, :] = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8], dtype=np.float32)
    cube[:, 2:, :] = np.asarray([4.0, 3.8, 3.6, 3.4, 3.2], dtype=np.float32)
    np.save(scene_root / "five_band.npy", cube)
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(scene_root / "preview.png")
    paint_mask = np.full((4, 4), 255, dtype=np.uint8)
    paint_mask_path = tmp_path / "paint_pred.png"
    Image.fromarray(paint_mask).save(paint_mask_path)

    exported = export_dual_pigment_analysis(
        scene_root=scene_root,
        output_root=tmp_path / "analysis",
        sample_record_path=sample_record,
        paint_mask_path=paint_mask_path,
        reference_scenes_root=reference_root,
    )

    assert len(exported) == 6
    assert (tmp_path / "analysis" / "dual_pigment_overlay.png").exists()
    assert (tmp_path / "analysis" / "dual_pigment_curves.png").exists()
    assert (tmp_path / "analysis" / "dual_pigment_summary.csv").exists()
    assert (tmp_path / "analysis" / "dual_pigment_summary.json").exists()
    log_text = (tmp_path / "analysis" / "dual_pigment_log.txt").read_text(encoding="utf-8")
    assert "cluster_1_label=" in log_text


