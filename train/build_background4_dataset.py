"""Build the isolated background4_v1 patch set from approved manual labels.

The workflow state is the authority for post-import samples.  This prevents
softcomp seeds, unsaved edits, and held samples from entering training.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from train.six_band_dataset import export_patch_dataset

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "train" / "camera_eval_workspace"
STATE = ROOT / "ui" / "analysis_workbench" / "workflow_state.json"
OUT = ROOT / "train" / "five_band_patches" / "background4_v1"
BASELINE_LEGACY_IDS = {f"SAMPLE_{number:03d}" for number in range(36, 50)}
# Of SAMPLE_050–059, only these three remain eligible. They still require
# manual save plus explicit adoption before patch export.
RETAINED_REVIEW_IDS = {"SAMPLE_050", "SAMPLE_053", "SAMPLE_055"}


def parse_metadata(scene_root: Path) -> dict[str, str]:
    path = scene_root / "metadata.txt"
    result: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
    return result


def split_for_group(scene_id: str, background: str, light: str) -> str:
    # Deterministic group-level assignment: no rotated/exported derivative can
    # cross a split.  Background/light are included so every stratum is spread.
    bucket = sum(map(ord, f"{background}|{light}|{scene_id}")) % 10
    return "test" if bucket == 0 else ("val" if bucket == 1 else "train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"samples": {}}
    selected: list[tuple[Path, str, str, str]] = []
    for scene_root in sorted(SCENES.glob("SAMPLE_*")):
        scene_id = scene_root.name
        if scene_id < "SAMPLE_036":
            continue
        record = state.get("samples", {}).get(scene_id, {})
        manual_approved = record.get("status") == "approved" and record.get("annotation_source") == "manual_from_review" and record.get("annotation_decision") == "accepted"
        if scene_id in BASELINE_LEGACY_IDS:
            eligible = True
        elif scene_id in RETAINED_REVIEW_IDS:
            eligible = manual_approved
        else:
            eligible = manual_approved and not ("SAMPLE_050" <= scene_id <= "SAMPLE_059")
        if not eligible:
            continue
        meta = parse_metadata(scene_root)
        historical_review = scene_id in RETAINED_REVIEW_IDS
        background = record.get("background_role") or meta.get("background_role") or ("代赭" if scene_id in BASELINE_LEGACY_IDS else ("历史审核" if historical_review else ""))
        light = str(record.get("light_level") or meta.get("light_level") or ("unknown" if historical_review else ""))
        if not background or (not historical_review and light not in {"1", "5", "10"}):
            raise ValueError(f"{scene_id} needs background_role and light_level (1/5/10) before export.")
        if not (scene_root / "masks" / "pigment.png").exists():
            raise FileNotFoundError(f"{scene_id} has no masks/pigment.png (0 ignore, 1朱砂, 2代赭, 3石青, 4石绿).")
        selected.append((scene_root, background, light, split_for_group(scene_id, background, light)))
    counts: dict[str, int] = defaultdict(int)
    manifest: list[dict[str, str]] = []
    for scene_root, background, light, split in selected:
        result = export_patch_dataset(scene_root, split, args.output_root, args.patch_size, args.stride)
        counts[split] += int(result["patches"])
        manifest.append({"scene_id": scene_root.name, "background_role": background, "light_level": light, "split": split})
    (args.output_root / "background4_manifest.json").write_text(json.dumps({"version_id": "background4_v1", "scenes": manifest, "patch_counts": counts}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"version_id": "background4_v1", "scene_count": len(selected), "patch_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
