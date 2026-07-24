"""Record and verify that background4_v1 does not alter softcomp assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "train" / "five_band_patches" / "background4_v1" / "protected_softcomp_baseline.json"
PROTECTED = {
    "softcomp_checkpoint": ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "retune_9_scene3647_v10_balanced_softcomp_4849_pollution4447_v1",
    "softcomp_predictions": ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "validation_v10_balanced_softcomp_5056_pollthr035",
    "softcomp_patches": ROOT / "train" / "five_band_patches" / "train_light_v10_balanced_4849_pollution4447",
}


def tree_digest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "files": 0, "sha256": None}
    digest = hashlib.sha256()
    count = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1
    return {"exists": True, "files": count, "sha256": digest.hexdigest()}


def snapshot() -> dict[str, object]:
    return {"version_id": "background4_v1", "protected": {name: tree_digest(path) for name, path in PROTECTED.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard softcomp files during background4_v1 work")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    current = snapshot()
    if args.write_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"action": "baseline_written", **current}, ensure_ascii=False))
        return
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    if current.get("protected") != expected.get("protected"):
        raise SystemExit("Protected softcomp assets changed; background4_v1 isolation check failed.")
    print(json.dumps({"action": "verified", **current}, ensure_ascii=False))


if __name__ == "__main__":
    main()
