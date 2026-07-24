import csv
from pathlib import Path

import numpy as np

from train.six_band_dataset import (
    detect_board_bbox_from_preview,
    main,
    export_camera_scene_from_images,
    export_cvat_annotations,
    ensure_envi_header_compatibility,
    create_six_band_preview,
    decode_cvat_rle,
    export_annotation_workspace,
    export_patch_dataset,
    extract_six_band_cube,
    iter_patch_windows,
    select_band_indices,
    select_latest_camera_band_images,
)


def test_select_band_indices_picks_nearest_requested_wavelengths() -> None:
    wavelengths = np.array([400.0, 405.0, 410.0, 445.0, 450.0, 455.0, 500.0, 550.0, 600.0, 650.0, 700.0])

    indices = select_band_indices(
        wavelengths=wavelengths,
        target_wavelengths=(450.0, 500.0, 550.0, 600.0, 650.0, 700.0),
    )

    assert indices == (4, 6, 7, 8, 9, 10)


def test_extract_six_band_cube_returns_requested_channels() -> None:
    cube = np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8)

    extracted = extract_six_band_cube(cube, (0, 1, 2, 3, 4, 5))

    assert extracted.shape == (4, 5, 6)
    assert np.array_equal(extracted[:, :, 0], cube[:, :, 0])
    assert np.array_equal(extracted[:, :, 5], cube[:, :, 5])


def test_create_six_band_preview_returns_uint8_rgb() -> None:
    cube = np.random.default_rng(0).random((16, 16, 6), dtype=np.float32)

    preview = create_six_band_preview(cube)

    assert preview.shape == (16, 16, 3)
    assert preview.dtype == np.uint8


def test_export_annotation_workspace_creates_preview_and_mask_templates(tmp_path: Path) -> None:
    export_annotation_workspace(
        scene_id="SCENE_001",
        preview_image=np.zeros((32, 32, 3), dtype=np.uint8),
        six_band_cube=np.zeros((32, 32, 6), dtype=np.float32),
        output_root=tmp_path,
    )

    assert (tmp_path / "SCENE_001" / "preview.png").exists()
    assert (tmp_path / "SCENE_001" / "six_band.npy").exists()
    assert (tmp_path / "SCENE_001" / "masks" / "paint.png").exists()
    assert (tmp_path / "SCENE_001" / "masks" / "pollution.png").exists()
    assert (tmp_path / "SCENE_001" / "masks" / "aging.png").exists()


def test_detect_board_bbox_from_preview_returns_largest_green_component_bbox() -> None:
    preview = np.zeros((10, 14, 3), dtype=np.uint8)
    preview[:, :] = np.array([240, 120, 240], dtype=np.uint8)
    preview[2:9, 3:11] = np.array([40, 160, 90], dtype=np.uint8)
    preview[0:2, 0:2] = np.array([30, 150, 80], dtype=np.uint8)

    bbox = detect_board_bbox_from_preview(preview)

    assert bbox == (3, 2, 11, 9)


def test_detect_board_bbox_from_preview_falls_back_to_bright_board_when_green_mask_is_tiny() -> None:
    preview = np.zeros((12, 16, 3), dtype=np.uint8)
    preview[:, :] = np.array([20, 20, 20], dtype=np.uint8)
    preview[1:11, 3:14] = np.array([235, 235, 225], dtype=np.uint8)
    preview[4:6, 7:9] = np.array([40, 170, 90], dtype=np.uint8)

    bbox = detect_board_bbox_from_preview(preview)

    assert bbox == (3, 1, 14, 11)


def test_iter_patch_windows_covers_scene_with_fixed_patch_size() -> None:
    windows = list(iter_patch_windows(width=512, height=512, patch_size=256, stride=256))

    assert windows == [
        (0, 0, 256, 256),
        (256, 0, 512, 256),
        (0, 256, 256, 512),
        (256, 256, 512, 512),
    ]


def test_iter_patch_windows_keeps_last_row_and_column_for_non_divisible_scene() -> None:
    windows = list(iter_patch_windows(width=1920, height=1200, patch_size=512, stride=512))

    assert windows == [
        (0, 0, 512, 512),
        (512, 0, 1024, 512),
        (1024, 0, 1536, 512),
        (1408, 0, 1920, 512),
        (0, 512, 512, 1024),
        (512, 512, 1024, 1024),
        (1024, 512, 1536, 1024),
        (1408, 512, 1920, 1024),
        (0, 688, 512, 1200),
        (512, 688, 1024, 1200),
        (1024, 688, 1536, 1200),
        (1408, 688, 1920, 1200),
    ]


def test_export_patch_dataset_writes_npy_and_three_masks(tmp_path: Path) -> None:
    scene_root = tmp_path / "SCENE_001"
    masks_root = scene_root / "masks"
    masks_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.zeros((32, 32, 5), dtype=np.float32))

    for mask_name in ("paint.png", "pollution.png", "aging.png"):
        mask = np.zeros((32, 32), dtype=np.uint8)
        from PIL import Image

        Image.fromarray(mask).save(masks_root / mask_name)

    counts = export_patch_dataset(
        scene_root=scene_root,
        split="train",
        output_root=tmp_path / "dataset",
        patch_size=16,
        stride=16,
    )

    assert counts["patches"] == 4
    assert (tmp_path / "dataset" / "train" / "images" / "SCENE_001_x0_y0.npy").exists()
    assert (tmp_path / "dataset" / "train" / "masks" / "SCENE_001_x0_y0" / "paint.png").exists()


def test_export_patch_dataset_keeps_edge_patch_when_scene_is_not_divisible(tmp_path: Path) -> None:
    scene_root = tmp_path / "SCENE_001"
    masks_root = scene_root / "masks"
    masks_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.zeros((1200, 1920, 5), dtype=np.float32))

    for mask_name in ("paint.png", "pollution.png", "aging.png"):
        mask = np.zeros((1200, 1920), dtype=np.uint8)
        from PIL import Image

        Image.fromarray(mask).save(masks_root / mask_name)

    counts = export_patch_dataset(
        scene_root=scene_root,
        split="train",
        output_root=tmp_path / "dataset",
        patch_size=512,
        stride=512,
    )

    assert counts["patches"] == 12
    assert (tmp_path / "dataset" / "train" / "images" / "SCENE_001_x1408_y688.npy").exists()


def test_export_patch_dataset_crops_to_board_bbox_before_tiling(tmp_path: Path) -> None:
    scene_root = tmp_path / "SCENE_001"
    masks_root = scene_root / "masks"
    masks_root.mkdir(parents=True)
    np.save(scene_root / "five_band.npy", np.zeros((8, 12, 5), dtype=np.float32))

    from PIL import Image

    preview = np.zeros((8, 12, 3), dtype=np.uint8)
    preview[:, :] = np.array([240, 120, 240], dtype=np.uint8)
    preview[1:7, 2:10] = np.array([40, 160, 90], dtype=np.uint8)
    Image.fromarray(preview).save(scene_root / "preview.png")

    for mask_name in ("paint.png", "pollution.png", "aging.png"):
        mask = np.zeros((8, 12), dtype=np.uint8)
        Image.fromarray(mask).save(masks_root / mask_name)

    counts = export_patch_dataset(
        scene_root=scene_root,
        split="train",
        output_root=tmp_path / "dataset",
        patch_size=4,
        stride=4,
    )

    assert counts["patches"] == 4
    assert (tmp_path / "dataset" / "train" / "images" / "SCENE_001_x2_y1.npy").exists()
    assert (tmp_path / "dataset" / "train" / "images" / "SCENE_001_x6_y3.npy").exists()
    assert not (tmp_path / "dataset" / "train" / "images" / "SCENE_001_x0_y0.npy").exists()

    patch = np.load(tmp_path / "dataset" / "train" / "images" / "SCENE_001_x2_y1.npy")
    assert patch.shape == (4, 4, 5)


def test_export_patch_dataset_accumulates_patch_index_across_scenes(tmp_path: Path) -> None:
    output_root = tmp_path / "dataset"
    for scene_name in ("SCENE_001", "SCENE_002"):
        scene_root = tmp_path / scene_name
        masks_root = scene_root / "masks"
        masks_root.mkdir(parents=True)
        np.save(scene_root / "five_band.npy", np.zeros((16, 16, 5), dtype=np.float32))
        for mask_name in ("paint.png", "pollution.png", "aging.png"):
            from PIL import Image

            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(masks_root / mask_name)
        export_patch_dataset(
            scene_root=scene_root,
            split="train",
            output_root=output_root,
            patch_size=16,
            stride=16,
        )

    rows = list(csv.DictReader((output_root / "train" / "patch_index.csv").open(encoding="utf-8-sig")))

    assert len(rows) == 2
    assert {row["scene_id"] for row in rows} == {"SCENE_001", "SCENE_002"}


def test_ensure_envi_header_compatibility_adds_missing_byte_order(tmp_path: Path) -> None:
    hdr_path = tmp_path / "scene.hdr"
    hdr_path.write_text(
        "ENVI\nsamples = 4\nlines = 4\nbands = 6\ndata type = 4\ninterleave = bsq\n",
        encoding="utf-8",
    )

    normalized_path = ensure_envi_header_compatibility(hdr_path)

    content = normalized_path.read_text(encoding="utf-8")
    assert "byte order = 0" in content


def _encode_bbox_mask(mask: np.ndarray) -> str:
    flat = mask.astype(np.uint8).reshape(-1)
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        if value == current:
            run += 1
            continue
        counts.append(run)
        current = int(value)
        run = 1
    counts.append(run)
    return ", ".join(str(count) for count in counts)


def test_decode_cvat_rle_restores_bbox_mask() -> None:
    bbox_mask = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )

    decoded = decode_cvat_rle(_encode_bbox_mask(bbox_mask), width=3, height=2)

    assert np.array_equal(decoded, bbox_mask * 255)


def test_export_cvat_annotations_writes_three_training_masks(tmp_path: Path) -> None:
    scene_root = tmp_path / "SCENE_001"
    export_annotation_workspace(
        scene_id="SCENE_001",
        preview_image=np.zeros((4, 4, 3), dtype=np.uint8),
        six_band_cube=np.zeros((4, 4, 6), dtype=np.float32),
        output_root=tmp_path,
    )

    pollution_bbox = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    aging_bbox = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    paint_bbox = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="preview.png" width="4" height="4">
    <mask label="pollution" rle="{_encode_bbox_mask(pollution_bbox)}" left="1" top="0" width="2" height="2" />
    <mask label="aging" rle="{_encode_bbox_mask(aging_bbox)}" left="0" top="2" width="2" height="2" />
    <mask label="paint" rle="{_encode_bbox_mask(paint_bbox)}" left="2" top="2" width="2" height="2" />
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    export_cvat_annotations(xml_path=xml_path, scene_root=scene_root)

    from PIL import Image

    pollution = np.asarray(Image.open(scene_root / "masks" / "pollution.png").convert("L"))
    aging = np.asarray(Image.open(scene_root / "masks" / "aging.png").convert("L"))
    paint = np.asarray(Image.open(scene_root / "masks" / "paint.png").convert("L"))

    assert pollution[0:2, 1:3].tolist() == [[255, 255], [0, 255]]
    assert aging[2:4, 0:2].tolist() == [[255, 0], [255, 255]]
    assert paint[2:4, 2:4].tolist() == [[0, 255], [255, 0]]


def test_export_camera_scene_from_images_writes_six_band_workspace(tmp_path: Path) -> None:
    image_paths: list[Path] = []
    for index in range(5):
        band = np.full((8, 10), fill_value=index * 10, dtype=np.uint8)
        image_path = tmp_path / f"band_{index}.bmp"
        from PIL import Image

        Image.fromarray(band, mode="L").save(image_path)
        image_paths.append(image_path)

    scene_root = export_camera_scene_from_images(
        scene_id="CAMERA_SCENE_001",
        image_paths=image_paths,
        output_root=tmp_path / "workspace",
    )

    cube = np.load(scene_root / "five_band.npy")
    assert cube.shape == (8, 10, 5)
    assert float(cube[0, 0, 0]) == 0.0
    assert np.isclose(float(cube[0, 0, 4]), (40.0 / 255.0) * 0.3)
    assert float(cube.max()) <= 0.3
    assert (scene_root / "preview.png").exists()
    assert (scene_root / "band_selection.txt").exists()
    band_text = (scene_root / "band_selection.txt").read_text(encoding="utf-8")
    assert "target_0=450.0" in band_text
    assert "target_1=550.0" in band_text
    assert "target_4=700.0" in band_text
    assert "500.0" not in band_text


def test_select_latest_camera_band_images_orders_newest_to_oldest_for_450_to_700(tmp_path: Path) -> None:
    from PIL import Image
    import os

    expected_names = [
        "capture_450.bmp",
        "capture_550.bmp",
        "capture_600.bmp",
        "capture_650.bmp",
        "capture_700.bmp",
    ]
    for index, name in enumerate(reversed(expected_names)):
        image_path = tmp_path / name
        Image.fromarray(np.full((4, 4), index, dtype=np.uint8), mode="L").save(image_path)
        timestamp = 1000.0 + index
        os.utime(image_path, (timestamp, timestamp))
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(tmp_path / "ignore.png")

    selected = select_latest_camera_band_images(tmp_path)

    assert [path.name for path in selected] == expected_names


def test_import_camera_images_cli_can_read_latest_directory_group(tmp_path: Path, capsys) -> None:
    from PIL import Image
    import os

    image_dir = tmp_path / "camera"
    image_dir.mkdir()
    for index, band_value in enumerate([70, 65, 60, 55, 45]):
        image_path = image_dir / f"capture_{index}.bmp"
        Image.fromarray(np.full((6, 7), band_value, dtype=np.uint8), mode="L").save(image_path)
        timestamp = 2000.0 + index
        os.utime(image_path, (timestamp, timestamp))

    main(
        [
            "import-camera-images",
            "--scene-id",
            "CAMERA_SCENE_LATEST",
            "--output-root",
            str(tmp_path / "workspace"),
            "--image-dir",
            str(image_dir),
        ]
    )

    output = capsys.readouterr().out
    assert "scene_root=" in output
    cube = np.load(tmp_path / "workspace" / "CAMERA_SCENE_LATEST" / "five_band.npy")
    assert np.isclose(float(cube[0, 0, 0]), (45.0 / 255.0) * 0.3)
    assert np.isclose(float(cube[0, 0, 4]), (70.0 / 255.0) * 0.3)


def test_import_annotations_cli_reports_scene_root(tmp_path: Path, capsys) -> None:
    export_annotation_workspace(
        scene_id="SCENE_001",
        preview_image=np.zeros((4, 4, 3), dtype=np.uint8),
        six_band_cube=np.zeros((4, 4, 6), dtype=np.float32),
        output_root=tmp_path,
    )
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="preview.png" width="4" height="4">
    <mask label="paint" rle="1, 3" left="0" top="0" width="2" height="2" />
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    main(
        [
            "import-annotations",
            "--annotations-xml",
            str(xml_path),
            "--scene-root",
            str(tmp_path / "SCENE_001"),
        ]
    )

    output = capsys.readouterr().out
    assert "paint_objects=1" in output
    assert f"scene_root={tmp_path / 'SCENE_001'}" in output


def test_import_camera_images_cli_reports_scene_root(tmp_path: Path, capsys) -> None:
    image_paths: list[Path] = []
    for index in range(5):
        band = np.full((6, 7), fill_value=index, dtype=np.uint8)
        image_path = tmp_path / f"capture_{index}.bmp"
        from PIL import Image

        Image.fromarray(band, mode="L").save(image_path)
        image_paths.append(image_path)

    main(
        [
            "import-camera-images",
            "--scene-id",
            "CAMERA_SCENE_002",
            "--output-root",
            str(tmp_path / "workspace"),
            "--image-paths",
            *[str(path) for path in image_paths],
        ]
    )

    output = capsys.readouterr().out
    assert "scene_root=" in output
    assert "channels=5" in output


def test_export_patches_cli_filters_requested_scene_ids(tmp_path: Path, capsys) -> None:
    input_root = tmp_path / "workspace"
    output_root = tmp_path / "patches"
    for scene_id in ("SAMPLE_001", "SAMPLE_002"):
        scene_root = input_root / scene_id
        masks_root = scene_root / "masks"
        masks_root.mkdir(parents=True)
        np.save(scene_root / "five_band.npy", np.zeros((16, 16, 5), dtype=np.float32))
        from PIL import Image

        for mask_name in ("paint.png", "pollution.png", "aging.png"):
            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(masks_root / mask_name)

    main(
        [
            "export-patches",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--split",
            "train",
            "--patch-size",
            "16",
            "--stride",
            "16",
            "--scene-ids",
            "SAMPLE_002",
        ]
    )

    output = capsys.readouterr().out
    assert "exported_patches=1" in output
    assert (output_root / "train" / "images" / "SAMPLE_002_x0_y0.npy").exists()
    assert not (output_root / "train" / "images" / "SAMPLE_001_x0_y0.npy").exists()


def test_detect_board_bbox_from_preview_keeps_full_board_when_left_half_is_darker() -> None:
    preview = np.zeros((12, 18, 3), dtype=np.uint8)
    preview[:, :] = np.array([24, 24, 24], dtype=np.uint8)
    preview[1:11, 2:16] = np.array([175, 175, 165], dtype=np.uint8)
    preview[1:11, 2:8] = np.array([102, 104, 100], dtype=np.uint8)
    preview[4:8, 10:14] = np.array([215, 215, 205], dtype=np.uint8)

    bbox = detect_board_bbox_from_preview(preview)

    assert bbox == (2, 1, 16, 11)
