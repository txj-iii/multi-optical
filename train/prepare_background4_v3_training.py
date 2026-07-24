from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "train" / "five_band_patches" / "background4_v1" / "train"
INCREMENTAL_ROOT = ROOT / "train" / "five_band_patches" / "background4_v3" / "train"
OUTPUT_PATH = ROOT / "train" / "five_band_patches" / "background4_v3" / "training_manifest.json"
REQUIRED_INCREMENTAL_SCENES = {f"SAMPLE_{value:03d}" for value in range(97, 120)}
PURE_BACKGROUND_SCENES = {
    "SAMPLE_036",
    "SAMPLE_037",
    "SAMPLE_038",
    "SAMPLE_039",
    "SAMPLE_063",
    "SAMPLE_065",
    "SAMPLE_090",
    "SAMPLE_091",
    "SAMPLE_092",
    "SAMPLE_093",
}


def scene_id_from_patch_name(patch_name: str) -> str:
    parts = patch_name.split("_")
    if len(parts) < 2 or parts[0] != "SAMPLE":
        raise ValueError(f"Unsupported patch name: {patch_name}")
    return "_".join(parts[:2])


def inspect_root(root: Path, root_id: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for image_path in sorted((root / "images").glob("*.npy")):
        patch_name = image_path.stem
        mask_root = root / "masks" / patch_name
        mask_paths = {name: mask_root / f"{name}.png" for name in ("paint", "pollution", "aging", "pigment")}
        missing = [str(path) for path in mask_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{patch_name} is missing masks: {missing}")
        paint = np.asarray(Image.open(mask_paths["paint"]).convert("L"), dtype=np.uint8) > 0
        pigment = np.asarray(Image.open(mask_paths["pigment"]).convert("L"), dtype=np.uint8)
        if not set(np.unique(pigment)).issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"{patch_name} has invalid pigment labels.")
        if np.any((pigment > 0) & ~paint):
            raise ValueError(f"{patch_name} has pigment pixels outside paint.")
        records.append(
            {
                "patch_name": patch_name,
                "scene_id": scene_id_from_patch_name(patch_name),
                "root_id": root_id,
                "image": f"images/{image_path.name}",
                "masks": {name: f"masks/{patch_name}/{name}.png" for name in mask_paths},
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the complete background4_v3 training allow-list.")
    parser.add_argument("--allow-missing-pending-aging", action="store_true")
    args = parser.parse_args()

    records = inspect_root(BASE_ROOT, "v2_frozen_base") + inspect_root(INCREMENTAL_ROOT, "v3_incremental")
    patch_names = [str(record["patch_name"]) for record in records]
    if len(patch_names) != len(set(patch_names)):
        raise ValueError("Patch names overlap between the frozen base and v3 incremental roots.")

    scene_ids = {str(record["scene_id"]) for record in records}
    missing_incremental = sorted(REQUIRED_INCREMENTAL_SCENES - scene_ids)
    if missing_incremental and not args.allow_missing_pending_aging:
        raise ValueError(
            "The v3 manifest cannot be frozen before the reviewed pure-aging scenes are exported: "
            + ", ".join(missing_incremental)
        )

    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")

    base_scene_ids = sorted(scene_id for scene_id in scene_ids if scene_id not in REQUIRED_INCREMENTAL_SCENES)
    incremental_scene_ids = sorted(scene_ids & REQUIRED_INCREMENTAL_SCENES)
    payload = {
        "version_id": "background4_v3",
        "purpose": "frozen_v2_base_plus_reviewed_v3_increment",
        "patch_roots": [str(BASE_ROOT), str(INCREMENTAL_ROOT)],
        "scene_ids": sorted(scene_ids),
        "pure_background_scene_ids": sorted(PURE_BACKGROUND_SCENES & scene_ids),
        "patch_count": len(records),
        "patch_counts_by_scene": dict(sorted(Counter(str(record["scene_id"]) for record in records).items())),
        "sampling_groups": {
            "v2_frozen_base": {"scene_ids": base_scene_ids, "target_fraction": 0.67},
            "v3_reviewed_increment": {"scene_ids": incremental_scene_ids, "target_fraction": 0.33},
        },
        "split_groups": [
            ["SAMPLE_097", "SAMPLE_098", "SAMPLE_099"],
            ["SAMPLE_100", "SAMPLE_101", "SAMPLE_102"],
            ["SAMPLE_103", "SAMPLE_104"],
            ["SAMPLE_105", "SAMPLE_106", "SAMPLE_107", "SAMPLE_108", "SAMPLE_109", "SAMPLE_110"],
            ["SAMPLE_111", "SAMPLE_112"],
            ["SAMPLE_113", "SAMPLE_114", "SAMPLE_115", "SAMPLE_116"],
            ["SAMPLE_117", "SAMPLE_118", "SAMPLE_119"],
        ],
        "records_sha256": digest.hexdigest(),
        "records": records,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUTPUT_PATH), "scene_count": len(scene_ids), "patch_count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
