from pathlib import Path

import numpy as np
from PIL import Image

from train.export_curves import (
    BAND_LABELS,
    compute_band_means,
    export_sample_curves,
    load_sample_pigments,
    normalize_curve_values,
    resolve_curve_style,
)


def test_load_sample_pigments_reads_markdown_table(tmp_path: Path) -> None:
    sample_record = tmp_path / "samples.md"
    sample_record.write_text(
        "# 样本记录规范\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | 石绿 |\n| SAMPLE_002 | 朱砂 |\n\n## 当前工作区对应关系\n\n| sample_id | 工作区目录 |\n| --- | --- |\n| SAMPLE_001 | train/camera_eval_workspace/SAMPLE_001 |\n",
        encoding="utf-8",
    )

    pigments = load_sample_pigments(sample_record)

    assert pigments == {"SAMPLE_001": "石绿", "SAMPLE_002": "朱砂"}


def test_compute_band_means_returns_five_band_region_means() -> None:
    five_band = np.stack(
        [
            np.full((2, 2), 1.0, dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
            np.full((2, 2), 3.0, dtype=np.float32),
            np.full((2, 2), 4.0, dtype=np.float32),
            np.full((2, 2), 5.0, dtype=np.float32),
        ],
        axis=-1,
    )
    mask = np.array([[255, 0], [255, 0]], dtype=np.uint8)

    means = compute_band_means(five_band, mask)

    assert means == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_normalize_curve_values_scales_by_max_value() -> None:
    normalized = normalize_curve_values([2.0, 4.0, 1.0, 3.0, 2.0])

    assert normalized == [0.5, 1.0, 0.25, 0.75, 0.5]


def test_resolve_curve_style_groups_same_pigment_into_same_color_family() -> None:
    color_a, label_a = resolve_curve_style("SAMPLE_001", "石绿")
    color_b, label_b = resolve_curve_style("SAMPLE_002", "石绿")
    color_c, label_c = resolve_curve_style("SAMPLE_004", "石青")

    assert label_a == "石绿"
    assert label_b == "石绿"
    assert color_a != color_b
    assert color_a[1] > color_a[0]
    assert color_b[1] > color_b[0]
    assert color_c[2] > color_c[1]


def test_export_sample_curves_writes_csv_png_and_skip_log(tmp_path: Path) -> None:
    scenes_root = tmp_path / "scenes"
    scenes_root.mkdir()
    sample_record = tmp_path / "samples.md"
    sample_record.write_text(
        "# 样本记录规范\n\n| sample_id | pigment |\n| --- | --- |\n| SAMPLE_001 | 石绿 |\n| SAMPLE_002 | 朱砂 |\n",
        encoding="utf-8",
    )

    scene_one = scenes_root / "SAMPLE_001"
    scene_one.mkdir()
    np.save(scene_one / "five_band.npy", np.ones((4, 4, len(BAND_LABELS)), dtype=np.float32))
    masks_one = scene_one / "masks"
    masks_one.mkdir()
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(masks_one / "paint.png")

    scene_two = scenes_root / "SAMPLE_002"
    scene_two.mkdir()
    np.save(scene_two / "five_band.npy", np.ones((4, 4, len(BAND_LABELS)), dtype=np.float32))
    masks_two = scene_two / "masks"
    masks_two.mkdir()
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(masks_two / "paint.png")

    rows, skipped = export_sample_curves(
        scenes_root=scenes_root,
        output_root=tmp_path / "curves",
        sample_record_path=sample_record,
        scene_ids=["SAMPLE_001", "SAMPLE_002"],
    )

    assert len(rows) == 1
    assert skipped == ["SAMPLE_002"]
    assert (tmp_path / "curves" / "curve_summary.csv").exists()
    assert (tmp_path / "curves" / "SAMPLE_001_curve.png").exists()
    assert (tmp_path / "curves" / "all_samples_curves.png").exists()
    assert (tmp_path / "curves" / "all_samples_curves_normalized.png").exists()
    log_text = (tmp_path / "curves" / "curve_export_log.txt").read_text(encoding="utf-8")
    assert "skipped=SAMPLE_002: empty paint mask" in log_text
