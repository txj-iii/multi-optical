"""Freeze and validate the first background4_v1 training patch set.

The manifest is intentionally an allow-list.  Training must receive it so
future UI imports cannot silently become training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "train" / "five_band_patches" / "background4_v1" / "train"
MANIFEST_PATH = ROOT / "train" / "five_band_patches" / "background4_v1" / "training_manifest.json"
SCENE_IDS = (
    *(f"SAMPLE_{number:03d}" for number in range(36, 40)),
    "SAMPLE_050", "SAMPLE_053", "SAMPLE_055",
    "SAMPLE_060", "SAMPLE_061", "SAMPLE_062", "SAMPLE_063", "SAMPLE_065",
    *(f"SAMPLE_{number:03d}" for number in range(66, 94)),
)
PURE_BACKGROUND_IDS = {
    "SAMPLE_036", "SAMPLE_037", "SAMPLE_038", "SAMPLE_039",
    "SAMPLE_063", "SAMPLE_065", "SAMPLE_090", "SAMPLE_091", "SAMPLE_092", "SAMPLE_093",
}


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _aggregate_sha256(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(patch_root: Path) -> dict[str, object]:
    image_root = patch_root / "images"
    mask_root = patch_root / "masks"
    records: list[dict[str, object]] = []
    errors: list[str] = []
    class_pixels: Counter[int] = Counter()
    scene_counts: Counter[str] = Counter()
    seen_scenes: set[str] = set()
    for image_path in sorted(image_root.glob("*.npy")):
        patch_name = image_path.stem
        scene_id = "_".join(patch_name.split("_")[:2])
        if scene_id not in SCENE_IDS:
            continue
        seen_scenes.add(scene_id)
        masks = {head: mask_root / patch_name / f"{head}.png" for head in ("paint", "pollution", "aging", "pigment")}
        missing = [head for head, path in masks.items() if not path.exists()]
        if missing:
            errors.append(f"{patch_name}: missing {','.join(missing)}")
            continue
        cube = np.load(image_path, mmap_mode="r")
        if cube.ndim != 3 or cube.shape[2] != 5:
            errors.append(f"{patch_name}: expected HxWx5 input, got {tuple(cube.shape)}")
            continue
        paint = _read_mask(masks["paint"])
        pigment = _read_mask(masks["pigment"])
        if paint.shape != pigment.shape:
            errors.append(f"{patch_name}: paint/pigment shape mismatch")
            continue
        invalid = (pigment > 4)
        outside = (pigment > 0) & (paint == 0)
        if invalid.any():
            errors.append(f"{patch_name}: pigment contains values outside 0..4")
        if outside.any():
            errors.append(f"{patch_name}: pigment exists outside paint")
        if scene_id in PURE_BACKGROUND_IDS and ((paint > 0).any() or (pigment > 0).any()):
            errors.append(f"{patch_name}: pure background has paint/pigment positives")
        values, counts = np.unique(pigment, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            if value:
                class_pixels[int(value)] += int(count)
        scene_counts[scene_id] += 1
        records.append({"patch_name": patch_name, "scene_id": scene_id, "image": f"images/{image_path.name}", "masks": {head: f"masks/{patch_name}/{head}.png" for head in masks}})
    missing_scenes = sorted(set(SCENE_IDS) - seen_scenes)
    if missing_scenes:
        errors.append("missing scenes: " + ", ".join(missing_scenes))
    if errors:
        raise ValueError("background4_v1 preflight failed:\n- " + "\n- ".join(errors))
    return {
        "version_id": "background4_v1",
        "purpose": "first_round_training_allow_list",
        "patch_root": str(patch_root),
        "scene_ids": list(SCENE_IDS),
        "pure_background_scene_ids": sorted(PURE_BACKGROUND_IDS),
        "patch_count": len(records),
        "patch_counts_by_scene": dict(sorted(scene_counts.items())),
        "pigment_pixel_counts": {str(key): value for key, value in sorted(class_pixels.items())},
        "records_sha256": _aggregate_sha256(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze background4_v1 training patches")
    parser.add_argument("--patch-root", type=Path, default=PATCH_ROOT)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    manifest = build_manifest(args.patch_root)
    if manifest["patch_count"] != 1500:
        raise ValueError(f"Expected exactly 1500 patches, got {manifest['patch_count']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("version_id", "patch_count", "records_sha256", "pigment_pixel_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
