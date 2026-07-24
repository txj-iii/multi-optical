from __future__ import annotations

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage
from spectral import envi

TARGET_WAVELENGTHS = (450.0, 500.0, 550.0, 600.0, 650.0, 700.0)
CAMERA_TARGET_WAVELENGTHS = (450.0, 550.0, 600.0, 650.0, 700.0)
MASK_NAMES = ("paint.png", "pollution.png", "aging.png")
CAMERA_IMPORT_TARGET_MAX = 0.3


def select_band_indices(
    wavelengths: np.ndarray,
    target_wavelengths: tuple[float, ...] = TARGET_WAVELENGTHS,
) -> tuple[int, ...]:
    indices: list[int] = []
    for target in target_wavelengths:
        indices.append(int(np.abs(wavelengths - target).argmin()))
    return tuple(indices)


def extract_six_band_cube(cube: np.ndarray, band_indices: tuple[int, ...]) -> np.ndarray:
    return cube[:, :, list(band_indices)].astype(np.float32)


def _stretch_to_uint8(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)
    low = float(np.percentile(band, 2))
    high = float(np.percentile(band, 98))
    if high <= low:
        return np.zeros_like(band, dtype=np.uint8)
    stretched = np.clip((band - low) / (high - low), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def create_six_band_preview(cube: np.ndarray) -> np.ndarray:
    if cube.shape[2] < 5:
        raise ValueError("Preview requires at least 5 channels.")
    red_index = min(4, cube.shape[2] - 1)
    green_index = min(2, cube.shape[2] - 1)
    blue_index = 0
    red = _stretch_to_uint8(cube[:, :, red_index])
    green = _stretch_to_uint8(cube[:, :, green_index])
    blue = _stretch_to_uint8(cube[:, :, blue_index])
    return np.stack([red, green, blue], axis=-1)


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    labels, component_count = ndimage.label(mask)
    if component_count == 0:
        raise ValueError("Could not detect board region from preview image.")

    best_slice = None
    best_area = -1
    for label_index, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        area = int((labels[component_slice] == label_index).sum())
        if area > best_area:
            best_area = area
            best_slice = component_slice

    if best_slice is None:
        raise ValueError("Could not determine board bounding box from connected components.")

    y_slice, x_slice = best_slice
    return (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)


def detect_board_bbox_from_preview(preview_image: np.ndarray) -> tuple[int, int, int, int]:
    if preview_image.ndim != 3 or preview_image.shape[2] != 3:
        raise ValueError(f"Expected RGB preview image, got shape {preview_image.shape}.")

    rgb = preview_image.astype(np.float32)
    height, width = preview_image.shape[:2]
    image_area = width * height

    green_board_mask = (
        (rgb[:, :, 1] > 80.0)
        & (rgb[:, :, 1] > rgb[:, :, 0] + 40.0)
        & (rgb[:, :, 1] > rgb[:, :, 2] + 30.0)
    )
    try:
        green_bbox = _largest_component_bbox(green_board_mask)
    except ValueError:
        green_bbox = None

    if green_bbox is not None:
        green_width = green_bbox[2] - green_bbox[0]
        green_height = green_bbox[3] - green_bbox[1]
        green_area = green_width * green_height
        # Classic green boards remain the first choice when the component is large enough.
        if (
            green_width >= int(width * 0.4)
            and green_height >= int(height * 0.4)
            and green_area >= int(image_area * 0.2)
        ):
            return green_bbox

    mean_intensity = rgb.mean(axis=2)
    channel_spread = rgb.max(axis=2) - rgb.min(axis=2)

    def _bright_bbox(min_intensity: float, max_spread: float) -> tuple[int, int, int, int]:
        return _largest_component_bbox((mean_intensity > min_intensity) & (channel_spread < max_spread))

    bright_bbox = _bright_bbox(120.0, 170.0)
    bright_width = bright_bbox[2] - bright_bbox[0]
    bright_height = bright_bbox[3] - bright_bbox[1]
    bright_cover = (bright_width * bright_height) / float(image_area)

    # For darker light-5 boards such as SAMPLE_054/055, the strict bright threshold only keeps the right-side glare.
    # If the candidate is too narrow or strongly right-shifted, relax the threshold to recover the full plasterboard.
    if bright_cover < 0.45 or bright_bbox[0] > int(width * 0.3):
        relaxed_bbox = _bright_bbox(90.0, 180.0)
        relaxed_width = relaxed_bbox[2] - relaxed_bbox[0]
        relaxed_height = relaxed_bbox[3] - relaxed_bbox[1]
        relaxed_cover = (relaxed_width * relaxed_height) / float(image_area)
        if relaxed_cover > bright_cover:
            return relaxed_bbox

    return bright_bbox


def _write_blank_masks(masks_root: Path, image_size: tuple[int, int]) -> None:
    masks_root.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    empty = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
    for mask_name in MASK_NAMES:
        empty.save(masks_root / mask_name)


def decode_cvat_rle(rle: str, width: int, height: int) -> np.ndarray:
    counts = [int(item.strip()) for item in rle.split(",") if item.strip()]
    flat = np.zeros(width * height, dtype=np.uint8)
    cursor = 0
    value = 0
    for count in counts:
        if value == 1:
            flat[cursor : cursor + count] = 255
        cursor += count
        value = 1 - value
    if cursor != width * height:
        raise ValueError(f"Decoded CVAT RLE length {cursor} does not match mask area {width * height}.")
    return flat.reshape(height, width)


def export_cvat_annotations(xml_path: Path, scene_root: Path, image_name: str = "preview.png") -> dict[str, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_node = next((node for node in root.findall("image") if node.attrib.get("name") == image_name), None)
    if image_node is None:
        raise ValueError(f"Could not find image '{image_name}' in {xml_path}.")

    image_width = int(image_node.attrib["width"])
    image_height = int(image_node.attrib["height"])
    masks_root = scene_root / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    combined_masks = {
        mask_name: np.zeros((image_height, image_width), dtype=np.uint8)
        for mask_name in MASK_NAMES
    }
    label_to_mask_name = {mask_name.removesuffix(".png"): mask_name for mask_name in MASK_NAMES}
    object_counts = {label: 0 for label in label_to_mask_name}

    for mask_node in image_node.findall("mask"):
        label = mask_node.attrib["label"]
        mask_name = label_to_mask_name.get(label)
        if mask_name is None:
            continue
        left = int(mask_node.attrib["left"])
        top = int(mask_node.attrib["top"])
        width = int(mask_node.attrib["width"])
        height = int(mask_node.attrib["height"])
        decoded = decode_cvat_rle(mask_node.attrib["rle"], width=width, height=height)
        existing = combined_masks[mask_name][top : top + height, left : left + width]
        combined_masks[mask_name][top : top + height, left : left + width] = np.maximum(existing, decoded)
        object_counts[label] += 1

    for mask_name, mask_array in combined_masks.items():
        Image.fromarray(mask_array).save(masks_root / mask_name)

    return object_counts


def export_annotation_workspace(
    scene_id: str,
    preview_image: np.ndarray,
    six_band_cube: np.ndarray,
    output_root: Path,
    cube_filename: str = "six_band.npy",
) -> Path:
    scene_root = output_root / scene_id
    scene_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview_image).save(scene_root / "preview.png")
    np.save(scene_root / cube_filename, six_band_cube.astype(np.float32))
    _write_blank_masks(scene_root / "masks", (preview_image.shape[1], preview_image.shape[0]))
    (scene_root / "metadata.txt").write_text(
        "\n".join(
            [
                f"scene_id={scene_id}",
                f"height={six_band_cube.shape[0]}",
                f"width={six_band_cube.shape[1]}",
                f"channels={six_band_cube.shape[2]}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return scene_root


def _load_single_band_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path)
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3:
        return array[:, :, 0].astype(np.float32)
    raise ValueError(f"Unsupported image shape for {image_path}: {array.shape}")


def normalize_camera_band(band: np.ndarray, target_max: float = CAMERA_IMPORT_TARGET_MAX) -> np.ndarray:
    band = band.astype(np.float32) / 255.0
    return band * target_max


def export_camera_scene_from_images(
    scene_id: str,
    image_paths: Sequence[Path],
    output_root: Path,
    target_wavelengths: tuple[float, ...] = CAMERA_TARGET_WAVELENGTHS,
) -> Path:
    if len(image_paths) != len(target_wavelengths):
        raise ValueError(
            f"Camera import requires {len(target_wavelengths)} images, got {len(image_paths)}."
        )

    bands = [normalize_camera_band(_load_single_band_image(path)) for path in image_paths]
    reference_shape = bands[0].shape
    if any(band.shape != reference_shape for band in bands):
        raise ValueError("All camera band images must have the same width and height.")

    six_band_cube = np.stack(bands, axis=-1).astype(np.float32)
    preview = create_six_band_preview(six_band_cube)
    scene_root = export_annotation_workspace(
        scene_id,
        preview,
        six_band_cube,
        output_root,
        cube_filename="five_band.npy",
    )
    (scene_root / "band_selection.txt").write_text(
        "\n".join(
            [
                *[f"target_{idx}={wave:.1f}" for idx, wave in enumerate(target_wavelengths)],
                *[
                    f"source_image_{idx}={Path(image_paths[idx]).name}"
                    for idx in range(len(image_paths))
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return scene_root


def select_latest_camera_band_images(image_dir: Path, count: int = len(CAMERA_TARGET_WAVELENGTHS)) -> list[Path]:
    bmp_paths = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".bmp"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if len(bmp_paths) < count:
        raise ValueError(f"Expected at least {count} BMP images under {image_dir}, got {len(bmp_paths)}.")
    return bmp_paths[:count]


def iter_patch_windows(
    width: int,
    height: int,
    patch_size: int,
    stride: int,
) -> Iterable[tuple[int, int, int, int]]:
    def _compute_positions(length: int) -> list[int]:
        if length <= patch_size:
            return [0]
        positions = list(range(0, length - patch_size + 1, stride))
        last_start = length - patch_size
        if positions[-1] != last_start:
            positions.append(last_start)
        return positions

    for top in _compute_positions(height):
        for left in _compute_positions(width):
            yield (left, top, left + patch_size, top + patch_size)


def export_patch_dataset(
    scene_root: Path,
    split: str,
    output_root: Path,
    patch_size: int,
    stride: int,
) -> dict[str, int]:
    cube_path = scene_root / "five_band.npy"
    if not cube_path.exists():
        cube_path = scene_root / "six_band.npy"
    six_band_cube = np.load(cube_path)
    masks_root = scene_root / "masks"
    masks = {
        mask_name: np.asarray(Image.open(masks_root / mask_name).convert("L"), dtype=np.uint8)
        for mask_name in MASK_NAMES
    }
    pigment_path = masks_root / "pigment.png"
    if pigment_path.exists():
        # 0=ignore/background; 1=朱砂, 2=代赭, 3=石青, 4=石绿.
        masks["pigment.png"] = np.asarray(Image.open(pigment_path).convert("L"), dtype=np.uint8)

    split_root = output_root / split
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, int | str]] = []

    count = 0
    scene_id = scene_root.name
    height, width, _ = six_band_cube.shape
    roi_left = 0
    roi_top = 0
    roi_right = width
    roi_bottom = height
    preview_path = scene_root / "preview.png"
    if preview_path.exists():
        try:
            roi_left, roi_top, roi_right, roi_bottom = detect_board_bbox_from_preview(
                np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.uint8)
            )
        except ValueError:
            roi_left = 0
            roi_top = 0
            roi_right = width
            roi_bottom = height

    roi_cube = six_band_cube[roi_top:roi_bottom, roi_left:roi_right, :]
    roi_masks = {
        mask_name: mask_array[roi_top:roi_bottom, roi_left:roi_right]
        for mask_name, mask_array in masks.items()
    }
    roi_height, roi_width, _ = roi_cube.shape
    for left, top, right, bottom in iter_patch_windows(roi_width, roi_height, patch_size, stride):
        absolute_left = roi_left + left
        absolute_top = roi_top + top
        patch_name = f"{scene_id}_x{absolute_left}_y{absolute_top}"
        np.save(images_dir / f"{patch_name}.npy", roi_cube[top:bottom, left:right, :].astype(np.float32))
        patch_mask_root = masks_dir / patch_name
        patch_mask_root.mkdir(parents=True, exist_ok=True)
        for mask_name, mask_array in roi_masks.items():
            Image.fromarray(mask_array[top:bottom, left:right]).save(patch_mask_root / mask_name)
        metadata_rows.append(
            {
                "patch_name": patch_name,
                "scene_id": scene_id,
                "left": absolute_left,
                "top": absolute_top,
                "right": roi_left + right,
                "bottom": roi_top + bottom,
                "roi_left": roi_left,
                "roi_top": roi_top,
                "roi_right": roi_right,
                "roi_bottom": roi_bottom,
            }
        )
        count += 1

    metadata_path = split_root / "patch_index.csv"
    existing_rows: list[dict[str, int | str]] = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("scene_id") != scene_id
            ]
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "patch_name",
                "scene_id",
                "left",
                "top",
                "right",
                "bottom",
                "roi_left",
                "roi_top",
                "roi_right",
                "roi_bottom",
            ),
        )
        writer.writeheader()
        writer.writerows(existing_rows + metadata_rows)

    return {"patches": count}


def _read_wavelengths(hdr_path: Path) -> np.ndarray:
    content = hdr_path.read_text(encoding="utf-8", errors="ignore")
    marker = "wavelength = {"
    start = content.index(marker) + len(marker)
    end = content.index("}", start)
    values = [float(item.strip()) for item in content[start:end].split(",") if item.strip()]
    return np.asarray(values, dtype=np.float32)


def ensure_envi_header_compatibility(hdr_path: Path) -> Path:
    content = hdr_path.read_text(encoding="utf-8", errors="ignore")
    if "byte order" in content.lower():
        return hdr_path
    normalized = content.rstrip() + "\nbyte order = 0\n"
    hdr_path.write_text(normalized, encoding="utf-8")
    return hdr_path


def _list_scene_ids(archive_path: Path) -> list[str]:
    scene_ids: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.namelist():
            if entry.endswith(".hdr"):
                scene_ids.add(Path(entry).stem)
    return sorted(scene_ids)


def export_six_band_scenes(
    archive_path: Path,
    output_root: Path,
    target_wavelengths: tuple[float, ...] = TARGET_WAVELENGTHS,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_root)

        for scene_id in _list_scene_ids(archive_path):
            hdr_path = temp_root / f"{scene_id}.hdr"
            img_path = temp_root / f"{scene_id}.img"
            if not hdr_path.exists() or not img_path.exists():
                continue
            ensure_envi_header_compatibility(hdr_path)
            wavelengths = _read_wavelengths(hdr_path)
            band_indices = select_band_indices(wavelengths, target_wavelengths)
            cube = np.asarray(envi.open(str(hdr_path), str(img_path)).load())
            six_band_cube = extract_six_band_cube(cube, band_indices)
            preview = create_six_band_preview(six_band_cube)
            scene_root = export_annotation_workspace(scene_id, preview, six_band_cube, output_root)
            (scene_root / "band_selection.txt").write_text(
                "\n".join(
                    [
                        *[f"target_{idx}={wave:.1f}" for idx, wave in enumerate(target_wavelengths)],
                        *[f"band_index_{idx}={band_indices[idx]}" for idx in range(len(band_indices))],
                        *[
                            f"source_wavelength_{idx}={float(wavelengths[band_indices[idx]]):.1f}"
                            for idx in range(len(band_indices))
                        ],
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            exported.append(scene_root)
    return exported


def _discover_scene_roots(input_root: Path, scene_ids: Sequence[str] | None = None) -> list[Path]:
    requested_scene_ids = set(scene_ids or [])
    scene_roots = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and ((path / "six_band.npy").exists() or (path / "five_band.npy").exists())
    )
    if not requested_scene_ids:
        return scene_roots
    return [path for path in scene_roots if path.name in requested_scene_ids]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare six-band dataset assets from raw ENVI hyperspectral scenes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_scenes_parser = subparsers.add_parser("export-scenes")
    export_scenes_parser.add_argument("--archive", type=str, required=True)
    export_scenes_parser.add_argument("--output-root", type=str, required=True)

    export_patches_parser = subparsers.add_parser("export-patches")
    export_patches_parser.add_argument("--input-root", type=str, required=True)
    export_patches_parser.add_argument("--output-root", type=str, required=True)
    export_patches_parser.add_argument("--split", type=str, default="train")
    export_patches_parser.add_argument("--patch-size", type=int, default=512)
    export_patches_parser.add_argument("--stride", type=int, default=512)
    export_patches_parser.add_argument("--scene-ids", type=str, nargs="*", default=None)

    import_annotations_parser = subparsers.add_parser("import-annotations")
    import_annotations_parser.add_argument("--annotations-xml", type=str, required=True)
    import_annotations_parser.add_argument("--scene-root", type=str, required=True)
    import_annotations_parser.add_argument("--image-name", type=str, default="preview.png")

    import_camera_images_parser = subparsers.add_parser("import-camera-images")
    import_camera_images_parser.add_argument("--scene-id", type=str, required=True)
    import_camera_images_parser.add_argument("--output-root", type=str, required=True)
    import_camera_images_parser.add_argument("--image-paths", type=str, nargs=5, default=None)
    import_camera_images_parser.add_argument("--image-dir", type=str, default=None)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "export-scenes":
        exported = export_six_band_scenes(Path(args.archive), Path(args.output_root))
        print(f"exported_scenes={len(exported)}")
        print(f"output_root={Path(args.output_root)}")
        return

    if args.command == "export-patches":
        input_root = Path(args.input_root)
        output_root = Path(args.output_root)
        total_patches = 0
        for scene_root in _discover_scene_roots(input_root, scene_ids=args.scene_ids):
            counts = export_patch_dataset(
                scene_root=scene_root,
                split=args.split,
                output_root=output_root,
                patch_size=args.patch_size,
                stride=args.stride,
            )
            total_patches += counts["patches"]
        print(f"exported_patches={total_patches}")
        return

    if args.command == "import-annotations":
        counts = export_cvat_annotations(
            xml_path=Path(args.annotations_xml),
            scene_root=Path(args.scene_root),
            image_name=args.image_name,
        )
        for label, count in counts.items():
            print(f"{label}_objects={count}")
        print(f"scene_root={Path(args.scene_root)}")
        return

    if args.command == "import-camera-images":
        if bool(args.image_paths) == bool(args.image_dir):
            raise ValueError("Pass exactly one of --image-paths or --image-dir.")
        image_paths = (
            [Path(path) for path in args.image_paths]
            if args.image_paths
            else select_latest_camera_band_images(Path(args.image_dir))
        )
        scene_root = export_camera_scene_from_images(
            scene_id=args.scene_id,
            image_paths=image_paths,
            output_root=Path(args.output_root),
        )
        print(f"scene_root={scene_root}")
        print("channels=5")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
