from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.predict import make_overlay_image


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose final predictions from per-head prediction roots.")
    parser.add_argument("--paint-root", type=str, required=True, help="Prediction root providing paint outputs.")
    parser.add_argument("--pollution-root", type=str, required=True, help="Prediction root providing pollution outputs.")
    parser.add_argument("--aging-root", type=str, required=True, help="Prediction root providing aging outputs.")
    parser.add_argument("--scenes-root", type=str, required=True, help="Scene root containing preview.png for each sample.")
    parser.add_argument("--output-root", type=str, required=True, help="Composed prediction output root.")
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Optional scene IDs to compose.")
    return parser.parse_args(argv)


def _resolve_scene_ids(scenes_root: Path, scene_ids: Sequence[str] | None) -> list[str]:
    if scene_ids:
        return list(scene_ids)
    return sorted(path.name for path in scenes_root.iterdir() if path.is_dir() and (path / "preview.png").exists())


def compose_predictions(
    *,
    paint_root: Path,
    pollution_root: Path,
    aging_root: Path,
    scenes_root: Path,
    output_root: Path,
    scene_ids: Sequence[str] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    color_map = {
        "paint": (255, 0, 0),
        "pollution": (255, 255, 0),
        "aging": (0, 128, 255),
    }
    source_roots = {
        "paint": paint_root,
        "pollution": pollution_root,
        "aging": aging_root,
    }
    for scene_id in _resolve_scene_ids(scenes_root, scene_ids):
        preview = np.asarray(Image.open(scenes_root / scene_id / "preview.png").convert("RGB"))
        target_dir = output_root / scene_id
        target_dir.mkdir(parents=True, exist_ok=True)
        combined_overlay = preview.copy()
        for head_name in ("paint", "pollution", "aging"):
            source_root = source_roots[head_name]
            mask_path = source_root / scene_id / f"{head_name}_pred.png"
            overlay_path = source_root / scene_id / f"{head_name}_overlay.png"
            if not mask_path.exists() or not overlay_path.exists():
                raise FileNotFoundError(f"Missing {head_name} outputs for {scene_id} under {source_root}.")
            target_mask_path = target_dir / mask_path.name
            target_overlay_path = target_dir / overlay_path.name
            if mask_path.resolve() != target_mask_path.resolve():
                shutil.copy2(mask_path, target_mask_path)
            if overlay_path.resolve() != target_overlay_path.resolve():
                shutil.copy2(overlay_path, target_overlay_path)
            mask_array = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            combined_overlay = make_overlay_image(combined_overlay, mask_array, color_map[head_name])
        Image.fromarray(combined_overlay).save(target_dir / "combined_overlay.png")


def main() -> None:
    args = parse_args()
    compose_predictions(
        paint_root=Path(args.paint_root),
        pollution_root=Path(args.pollution_root),
        aging_root=Path(args.aging_root),
        scenes_root=Path(args.scenes_root),
        output_root=Path(args.output_root),
        scene_ids=args.scene_ids,
    )
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
