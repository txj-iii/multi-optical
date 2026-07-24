from pathlib import Path

import numpy as np
from PIL import Image

from train.compose_predictions import compose_predictions, parse_args


def test_parse_args_accepts_per_head_roots() -> None:
    args = parse_args(
        [
            "--paint-root",
            "C:/demo/paint",
            "--pollution-root",
            "C:/demo/pollution",
            "--aging-root",
            "C:/demo/aging",
            "--scenes-root",
            "C:/demo/scenes",
            "--output-root",
            "C:/demo/output",
        ]
    )

    assert args.paint_root == "C:/demo/paint"
    assert args.pollution_root == "C:/demo/pollution"
    assert args.aging_root == "C:/demo/aging"


def test_compose_uses_requested_head_sources(tmp_path: Path) -> None:
    scene_root = tmp_path / "scenes" / "SAMPLE_001"
    scene_root.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(scene_root / "preview.png")

    expected_pixels = {
        "paint": 255,
        "pollution": 128,
        "aging": 64,
    }
    for head_name, root_name in (
        ("paint", "paint_root"),
        ("pollution", "pollution_root"),
        ("aging", "aging_root"),
    ):
        target = tmp_path / root_name / "SAMPLE_001"
        target.mkdir(parents=True)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = expected_pixels[head_name]
        Image.fromarray(mask).save(target / f"{head_name}_pred.png")
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(target / f"{head_name}_overlay.png")

    compose_predictions(
        paint_root=tmp_path / "paint_root",
        pollution_root=tmp_path / "pollution_root",
        aging_root=tmp_path / "aging_root",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "output",
        scene_ids=["SAMPLE_001"],
    )

    assert np.asarray(Image.open(tmp_path / "output" / "SAMPLE_001" / "paint_pred.png"))[1, 1] == 255
    assert np.asarray(Image.open(tmp_path / "output" / "SAMPLE_001" / "pollution_pred.png"))[1, 1] == 128
    assert np.asarray(Image.open(tmp_path / "output" / "SAMPLE_001" / "aging_pred.png"))[1, 1] == 64


def test_compose_skips_copy_when_source_and_target_are_same_file(tmp_path: Path) -> None:
    scene_root = tmp_path / "scenes" / "SAMPLE_001"
    scene_root.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(scene_root / "preview.png")

    output_scene = tmp_path / "output" / "SAMPLE_001"
    output_scene.mkdir(parents=True)

    paint_root = tmp_path / "paint_root" / "SAMPLE_001"
    paint_root.mkdir(parents=True)
    paint_mask = np.zeros((4, 4), dtype=np.uint8)
    paint_mask[0:2, 0:2] = 255
    Image.fromarray(paint_mask).save(paint_root / "paint_pred.png")
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(paint_root / "paint_overlay.png")

    for head_name, pixel in (("pollution", 128), ("aging", 64)):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[2:4, 2:4] = pixel
        Image.fromarray(mask).save(output_scene / f"{head_name}_pred.png")
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(output_scene / f"{head_name}_overlay.png")

    compose_predictions(
        paint_root=tmp_path / "paint_root",
        pollution_root=tmp_path / "output",
        aging_root=tmp_path / "output",
        scenes_root=tmp_path / "scenes",
        output_root=tmp_path / "output",
        scene_ids=["SAMPLE_001"],
    )

    assert np.asarray(Image.open(output_scene / "paint_pred.png"))[0, 0] == 255
    assert np.asarray(Image.open(output_scene / "pollution_pred.png"))[2, 2] == 128
    assert np.asarray(Image.open(output_scene / "aging_pred.png"))[2, 2] == 64
