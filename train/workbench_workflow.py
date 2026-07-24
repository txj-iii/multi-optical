from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.analysis_workbench import export_workbench_manifest
from train.compose_predictions import compose_predictions
from train.predict import export_five_band_predictions, make_overlay_image
from train.six_band_dataset import detect_board_bbox_from_preview, export_camera_scene_from_images, export_patch_dataset, select_latest_camera_band_images

IMAGE_ROOT = Path(os.environ.get("CAMERA_IMAGE_ROOT", r"D:\Software\HuaTengVision\Image"))
SCENES_ROOT = PROJECT_ROOT / "train" / "camera_eval_workspace"
STATE_PATH = PROJECT_ROOT / "ui" / "analysis_workbench" / "workflow_state.json"
MANIFEST_PATH = PROJECT_ROOT / "ui" / "analysis_workbench" / "workbench_manifest.json"
VERSION_PROVENANCE_PATH = PROJECT_ROOT / "ui" / "analysis_workbench" / "workbench_version_provenance.json"
SAMPLE_RECORD_PATH = PROJECT_ROOT / "readme" / "样本记录规范.md"
MAIN_PREDICTION_ROOT = PROJECT_ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "validation_v10_balanced_softcomp_5056_pollthr035"
MAIN_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "retune_9_scene3647_pollutionshape_v2" / "vnir_multitask_bootstrap_latest.pt"
VALIDATION_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "retune_9_scene3647_v10_balanced_softcomp_4849_pollution4447_v1" / "vnir_multitask_bootstrap_latest.pt"
PAINT_OVERRIDE_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "retune_9_scene3647_agingmix_4849_v1" / "vnir_multitask_bootstrap_latest.pt"
PAINT_OVERRIDE_PREDICTION_ROOT = PROJECT_ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "retune_9_scene3647_agingmix_4849_v1_selected"
AGING_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "retune_9_scene3649_agingonly_v5_edgefix_light" / "aging_only_finetune_latest.pt"
REVIEW_PREDICTION_ROOT = PROJECT_ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "ui_workbench_review"
BACKGROUND4_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "background4_v1" / "background4_v1.pt"
BACKGROUND4_PREDICTION_ROOT = PROJECT_ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "background4_v1"
BACKGROUND4_V2_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "background4_v2" / "background4_v2.pt"
BACKGROUND4_V2_PREDICTION_ROOT = PROJECT_ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "background4_v2"
BACKGROUND4_V3_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "background4_v3" / "background4_v3.pt"
BACKGROUND4_V3_PREDICTION_ROOT = PROJECT_ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "background4_v3"
BACKGROUND4_V3_AGINGFIX_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / "background4_v3_agingfix_v1" / "background4_v3_agingfix_v1_best.pt"
BACKGROUND4_V3_AGINGFIX_PREDICTION_ROOT = PROJECT_ROOT / "train" / "experiments" / "five_band_predictions" / "task_specific" / "background4_v3_agingfix_v1_best"
PAINT_OVERRIDE_SCENE_START = 48
TMP_ROOT = PROJECT_ROOT / "tmp" / "workbench_workflow"
MAIN_VERSION_ID = "validation_v10_balanced_softcomp_5056_pollthr035"
MAIN_VERSION_LABEL = "当前验证版 validation_v10_balanced_softcomp_5056_pollthr035"
STATUS_LABELS = {
    "awaiting_background": "待选择背景",
    "pending_review": "\u5f85\u5ba1\u6838",
    "approved": "\u5df2\u91c7\u7528",
    "held": "暂不采用",
    "error": "执行失败",
}
BACKGROUND_ROLES = ("代赭", "石青", "石绿", "朱砂")
LIGHT_LEVELS = ("1", "5", "10")
BACKGROUND4_PATCH_ROOT = PROJECT_ROOT / "train" / "five_band_patches" / "background4_v1"
BACKGROUND4_V3_PATCH_ROOT = PROJECT_ROOT / "train" / "five_band_patches" / "background4_v3"
BACKGROUND4_IMPORTS = (
    ("data1/A/5", "代赭", "A", "5", "mother"),
    ("data2/a/5", "石青", "A", "5", "mother"),
    ("data3/a/5", "石绿", "A", "5", "mother"),
    ("data4/背景/5", "朱砂", "background", "5", "mother"),
)
BACKGROUND4_PURE_BACKGROUNDS = (
    ("data2/背景/5", "石青", "5"),
    ("data2/背景/10", "石青", "10"),
    ("data3/背景/5", "石绿", "5"),
    ("data3/背景/10", "石绿", "10"),
)
DATA_ROOT = IMAGE_ROOT / "data"
HEAD_COLORS = {
    "paint": (255, 0, 0),
    "pollution": (255, 255, 0),
    "aging": (0, 128, 255),
}
AUDIT_ROOT_NAME = "annotation_audit"
AUDIT_DIFF_COLORS = {
    "missing": (28, 176, 93),
    "overmark": (196, 64, 196),
}



def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _annotation_rule_source() -> str | None:
    return str(SAMPLE_RECORD_PATH) if SAMPLE_RECORD_PATH.exists() else None


def _default_annotation_fields() -> dict[str, Any]:
    return {
        "annotation_source": "auto_seed_from_model",
        "annotation_source_label": "自动起标（validation 预测写入 masks 作为可编辑草稿）",
        "annotation_rule_source": _annotation_rule_source(),
        "annotation_rule_label": "validation 预测已写入 masks 作为可编辑草稿；标注语义按样本记录规范复核",
        "annotation_decision": "pending",
    }


def _write_scene_metadata(scene_id: str, **values: str) -> None:
    """Update workflow metadata without replacing camera-exported metadata."""
    path = SCENES_ROOT / scene_id / "metadata.txt"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    keys = set(values)
    lines = [line for line in lines if line.split("=", 1)[0].strip() not in keys]
    lines.extend(f"{key}={value}" for key, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rule_seed_annotation_fields() -> dict[str, Any]:
    return {
        "annotation_source": "rule_based_seed",
        "annotation_source_label": "历史半自动起标（板内 ROI + 颜色/亮度/形状规则）",
        "annotation_rule_source": _annotation_rule_source(),
        "annotation_rule_label": "历史规则起标草稿；当前主流程以 review 预测写入 masks 为可编辑底稿",
        "annotation_decision": "pending",
    }


def _sanitize_workflow_record(scene_id: str, record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    sanitized = dict(record)
    changed = False

    for key in ("patch_root", "train_dir", "last_train_stdout", "last_predict_stdout"):
        if key in sanitized:
            sanitized.pop(key, None)
            changed = True

    prediction_root_text = sanitized.get("prediction_root")
    if isinstance(prediction_root_text, str) and "ui_workbench_candidate" in prediction_root_text.replace("\\", "/"):
        sanitized["prediction_root"] = str(MAIN_PREDICTION_ROOT)
        changed = True

    if sanitized.get("status") == "approved" and sanitized.get("annotation_decision") != "accepted":
        sanitized["annotation_decision"] = "accepted"
        changed = True

    if sanitized.get("status") == "approved" and sanitized.get("stage") == "annotation_saved":
        sanitized["stage"] = "annotation_approved"
        changed = True

    if sanitized.get("stage") == "candidate_ready":
        approved = sanitized.get("status") == "approved" or sanitized.get("annotation_decision") == "accepted"
        sanitized["stage"] = "annotation_approved" if approved else "annotation_saved"
        changed = True

    return sanitized, changed


def load_workflow_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {"samples": {}, "updated_at": None}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    sample_records = payload.setdefault("samples", {})
    changed = False
    for scene_id, record in list(sample_records.items()):
        if not isinstance(record, dict):
            continue
        sanitized, record_changed = _sanitize_workflow_record(scene_id, record)
        if record_changed:
            sample_records[scene_id] = sanitized
            changed = True
    if changed:
        payload["updated_at"] = _now_iso()
        _ensure_parent(state_path)
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def save_workflow_state(state: dict[str, Any], state_path: Path = STATE_PATH) -> dict[str, Any]:
    state["updated_at"] = _now_iso()
    _ensure_parent(state_path)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def next_sample_id(scenes_root: Path = SCENES_ROOT) -> str:
    max_index = 0
    if scenes_root.exists():
        for path in scenes_root.iterdir():
            if not path.is_dir() or not path.name.startswith("SAMPLE_"):
                continue
            suffix = path.name.split("_", 1)[1]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return f"SAMPLE_{max_index + 1:03d}"


def _latest_bmps_in_directory(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".bmp"],
        key=lambda item: (item.stat().st_mtime, item.name),
        reverse=True,
    )


def discover_latest_capture_group(image_root: Path = IMAGE_ROOT, count: int = 5) -> list[Path]:
    candidate_dirs: list[tuple[float, list[Path]]] = []
    if image_root.exists():
        # Include the camera root itself: fresh captures are written directly
        # here, while archived groups live in subdirectories.
        for directory in (image_root, *image_root.rglob("*")):
            if not directory.is_dir():
                continue
            bmps = _latest_bmps_in_directory(directory)
            if len(bmps) >= count:
                candidate_dirs.append((bmps[0].stat().st_mtime, bmps[:count]))
    if candidate_dirs:
        candidate_dirs.sort(key=lambda item: item[0], reverse=True)
        return candidate_dirs[0][1]

    recursive_bmps = sorted(
        [path for path in image_root.rglob("*.bmp") if path.is_file()],
        key=lambda item: (item.stat().st_mtime, item.name),
        reverse=True,
    )
    if len(recursive_bmps) < count:
        raise ValueError(f"Expected at least {count} BMP images under {image_root}, got {len(recursive_bmps)}.")
    return recursive_bmps[:count]


def build_manifest_scene_selection(
    base_scene_ids: Sequence[str],
    workflow_records: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Path], dict[str, Path]]:
    scene_ids = list(base_scene_ids)
    base_scene_set = set(base_scene_ids)
    overrides: dict[str, Path] = {}
    pigment_overrides: dict[str, Path] = {}
    for scene_id in sorted(workflow_records):
        record = workflow_records[scene_id]
        prediction_root = record.get("prediction_root")
        pigment_root = record.get("pigment_root")
        is_base_scene = scene_id in base_scene_set
        display_mode = str(record.get("display_prediction_mode") or ("validation" if is_base_scene else "workflow"))
        if not is_base_scene and scene_id not in scene_ids:
            scene_ids.append(scene_id)
        use_workflow_prediction = bool(prediction_root) and (not is_base_scene or display_mode == "workflow")
        if use_workflow_prediction:
            overrides[scene_id] = Path(str(prediction_root))
        if pigment_root:
            pigment_overrides[scene_id] = Path(str(pigment_root))
    return scene_ids, overrides, pigment_overrides


def _discover_base_scene_ids(prediction_root: Path = MAIN_PREDICTION_ROOT) -> list[str]:
    if not prediction_root.exists():
        return []
    return sorted(
        path.name
        for path in prediction_root.iterdir()
        if path.is_dir() and (path / "combined_overlay.png").exists()
    )


def _load_version_provenance() -> dict[str, Any] | None:
    if not VERSION_PROVENANCE_PATH.exists():
        return None
    return json.loads(VERSION_PROVENANCE_PATH.read_text(encoding="utf-8"))


def _default_scene_pigment_roots() -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for scene_number in (48, 49):
        roots[f"SAMPLE_{scene_number:03d}"] = PAINT_OVERRIDE_PREDICTION_ROOT
    return roots


def refresh_manifest(state: dict[str, Any] | None = None) -> dict[str, Any]:
    workflow_state = state or load_workflow_state()
    _backfill_prediction_seed_annotations(workflow_state, scenes_root=SCENES_ROOT)
    base_scene_ids = _discover_base_scene_ids()
    scene_ids, overrides, pigment_overrides = build_manifest_scene_selection(base_scene_ids, workflow_state.get("samples", {}))
    pigment_roots = _default_scene_pigment_roots()
    pigment_roots.update(pigment_overrides)
    manifest = export_workbench_manifest(
        scenes_root=SCENES_ROOT,
        prediction_root=MAIN_PREDICTION_ROOT,
        output_path=MANIFEST_PATH,
        scene_ids=scene_ids,
        sample_record_path=SAMPLE_RECORD_PATH if SAMPLE_RECORD_PATH.exists() else None,
        scene_prediction_roots=overrides,
        scene_pigment_roots=pigment_roots,
        version_id=MAIN_VERSION_ID,
        version_label=MAIN_VERSION_LABEL,
        version_provenance=_load_version_provenance(),
        additional_versions=[],
    )
    return manifest


def _copy_prediction_masks_to_scene(scene_id: str, prediction_root: Path, scenes_root: Path = SCENES_ROOT) -> None:
    scene_root = scenes_root / scene_id
    masks_root = scene_root / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)
    for head_name in ("paint", "pollution", "aging"):
        shutil.copy2(prediction_root / scene_id / f"{head_name}_pred.png", masks_root / f"{head_name}.png")
    pigment_prediction = prediction_root / scene_id / "pigment_pred.png"
    if pigment_prediction.exists():
        # Dense pigment labels are already gated by predicted paint at export.
        # Seed them as editable labels so the UI does not discard a useful
        # pixel-level start point.  This helper is never called for a saved or
        # accepted annotation without the explicit reset flow.
        shutil.copy2(pigment_prediction, masks_root / "pigment.png")


def _render_annotation_overlay(scene_id: str, scenes_root: Path = SCENES_ROOT) -> Path:
    scene_root = scenes_root / scene_id
    preview = np.asarray(Image.open(scene_root / "preview.png").convert("RGB"), dtype=np.uint8)
    combined = preview.copy()
    for head_name in ("paint", "pollution", "aging"):
        mask = np.asarray(Image.open(scene_root / "masks" / f"{head_name}.png").convert("L"), dtype=np.uint8)
        combined = make_overlay_image(combined, mask, HEAD_COLORS[head_name])
    output_path = scene_root / "annotation_overlay.png"
    Image.fromarray(combined).save(output_path)
    return output_path


def _seed_scene_annotations_from_prediction(scene_id: str, prediction_root: Path, scenes_root: Path = SCENES_ROOT) -> dict[str, Any]:
    _copy_prediction_masks_to_scene(scene_id, prediction_root, scenes_root=scenes_root)
    overlay_path = _render_annotation_overlay(scene_id, scenes_root=scenes_root)
    positive_pixels: dict[str, int] = {}
    for head_name in ("paint", "pollution", "aging"):
        mask_path = scenes_root / scene_id / "masks" / f"{head_name}.png"
        positive_pixels[head_name] = int(_load_binary_mask(mask_path).sum()) if mask_path.exists() else 0
    return {
        "scene_id": scene_id,
        "annotation_overlay": str(overlay_path),
        "positive_pixels": positive_pixels,
    }


def _scene_annotation_has_positive_pixels(scene_id: str, scenes_root: Path = SCENES_ROOT) -> bool:
    masks_root = scenes_root / scene_id / "masks"
    if not masks_root.exists():
        return False
    for head_name in ("paint", "pollution", "aging"):
        mask_path = masks_root / f"{head_name}.png"
        if not mask_path.exists():
            return False
        if bool(_load_binary_mask(mask_path).any()):
            return True
    return False


def _should_backfill_prediction_annotation(record: dict[str, Any]) -> bool:
    annotation_source = str(record.get("annotation_source") or "")
    annotation_decision = str(record.get("annotation_decision") or "")
    if annotation_source == "manual_from_review":
        return False
    if annotation_decision in {"saved", "accepted"}:
        return False
    if not annotation_source and not annotation_decision:
        return False
    return bool(record.get("prediction_root"))


def _backfill_prediction_seed_annotations(workflow_state: dict[str, Any], scenes_root: Path = SCENES_ROOT) -> None:
    for scene_id, record in workflow_state.get("samples", {}).items():
        if not _should_backfill_prediction_annotation(record):
            continue
        if _scene_annotation_has_positive_pixels(scene_id, scenes_root=scenes_root):
            if str(record.get("annotation_source") or "") != "auto_seed_from_model":
                record.update(_default_annotation_fields())
            continue
        prediction_root_text = record.get("prediction_root")
        if not prediction_root_text:
            continue
        prediction_root = Path(str(prediction_root_text))
        scene_prediction_root = prediction_root / scene_id
        if not scene_prediction_root.exists():
            continue
        _seed_scene_annotations_from_prediction(scene_id, prediction_root, scenes_root=scenes_root)
        record.update(_default_annotation_fields())


def _load_binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def _save_binary_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8)) * 255).save(path)


def _decode_mask_payload(data: str) -> np.ndarray:
    if not isinstance(data, str) or not data:
        raise ValueError("Mask payload must be a non-empty data URL.")
    if "," not in data:
        raise ValueError("Mask payload is not a valid data URL.")
    header, encoded = data.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Mask payload must be base64-encoded.")
    raw = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(image, dtype=np.uint8) > 127


def _decode_label_payload(data: str) -> np.ndarray:
    if not isinstance(data, str) or "," not in data:
        raise ValueError("Pigment label payload must be a base64 image data URL.")
    header, encoded = data.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Pigment label payload must be base64-encoded.")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L"), dtype=np.uint8)


def _load_scene_preview_shape(scene_id: str, scenes_root: Path | None = None) -> tuple[int, int]:
    root = scenes_root or SCENES_ROOT
    preview = np.asarray(Image.open(root / scene_id / "preview.png").convert("RGB"), dtype=np.uint8)
    return int(preview.shape[0]), int(preview.shape[1])


def _validate_mask_shape(scene_id: str, mask: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if mask.shape != expected_shape:
        raise ValueError(
            f"{scene_id} mask shape mismatch: expected {expected_shape}, got {mask.shape}."
        )

def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return mask
    return ndimage.binary_fill_holes(mask).astype(bool)


def _morphology_cleanup(mask: np.ndarray, *, close_iterations: int = 1, open_iterations: int = 0) -> np.ndarray:
    if not bool(mask.any()):
        return mask
    structure = np.ones((3, 3), dtype=bool)
    cleaned = mask.astype(bool)
    if close_iterations > 0:
        cleaned = ndimage.binary_closing(cleaned, structure=structure, iterations=close_iterations)
    if open_iterations > 0:
        cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=open_iterations)
    return cleaned.astype(bool)


def _write_seed_masks(scene_root: Path, masks: dict[str, np.ndarray]) -> None:
    masks_root = scene_root / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)
    for head_name in ("paint", "pollution", "aging"):
        _save_binary_mask(masks_root / f"{head_name}.png", masks[head_name])


def _build_rule_based_seed_masks(
    preview_array: np.ndarray,
    *,
    five_band: np.ndarray | None = None,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, np.ndarray]:
    if preview_array.ndim != 3 or preview_array.shape[2] != 3:
        raise ValueError(f"Expected RGB preview array, got shape {preview_array.shape}.")

    height, width = preview_array.shape[:2]
    if roi_bbox is None:
        roi_bbox = detect_board_bbox_from_preview(preview_array)
    roi_left, roi_top, roi_right, roi_bottom = roi_bbox
    roi_left = max(0, min(width, int(roi_left)))
    roi_right = max(roi_left + 1, min(width, int(roi_right)))
    roi_top = max(0, min(height, int(roi_top)))
    roi_bottom = max(roi_top + 1, min(height, int(roi_bottom)))

    roi_preview = preview_array[roi_top:roi_bottom, roi_left:roi_right].astype(np.float32)
    if five_band is not None:
        roi_five_band = five_band[roi_top:roi_bottom, roi_left:roi_right].astype(np.float32)
        band_std = roi_five_band.std(axis=2)
    else:
        band_std = np.zeros(roi_preview.shape[:2], dtype=np.float32)

    mean_intensity = roi_preview.mean(axis=2)
    spread = roi_preview.max(axis=2) - roi_preview.min(axis=2)
    band = max(2, min(roi_preview.shape[0], roi_preview.shape[1]) // 24)
    border_mask = np.zeros(roi_preview.shape[:2], dtype=bool)
    border_mask[:band, :] = True
    border_mask[-band:, :] = True
    border_mask[:, :band] = True
    border_mask[:, -band:] = True
    border_pixels = roi_preview[border_mask]
    if border_pixels.size == 0:
        border_pixels = roi_preview.reshape(-1, 3)
    background_rgb = np.median(border_pixels, axis=0)
    background_distance = np.linalg.norm(roi_preview - background_rgb, axis=2)
    background_spread = float(np.median(spread[border_mask])) if bool(border_mask.any()) else float(np.median(spread))

    deviation_threshold = max(float(np.percentile(background_distance, 78)), 26.0)
    spread_threshold = max(float(np.percentile(spread, 72)), background_spread + 18.0, 36.0)
    bright_threshold = max(float(np.percentile(mean_intensity, 88)), 190.0)
    low_chroma_threshold = max(float(np.percentile(spread, 55)), 22.0)
    spectral_threshold = max(float(np.percentile(band_std, 80)), 0.015) if band_std.size else 0.015

    bright_low_chroma = (mean_intensity >= bright_threshold) & (spread <= low_chroma_threshold)
    foreground = (
        ((background_distance >= deviation_threshold) & (spread >= background_spread + 6.0))
        | (spread >= spread_threshold)
        | ((band_std >= spectral_threshold) & (spread >= background_spread + 4.0))
        | bright_low_chroma
    )
    foreground &= mean_intensity >= 32.0
    foreground = _fill_mask_holes(_morphology_cleanup(foreground, close_iterations=1, open_iterations=0))
    foreground[:1, :] = False
    foreground[-1:, :] = False
    foreground[:, :1] = False
    foreground[:, -1:] = False

    roi_masks = {head_name: np.zeros(roi_preview.shape[:2], dtype=bool) for head_name in ("paint", "pollution", "aging")}
    roi_area = max(int(foreground.shape[0] * foreground.shape[1]), 1)

    for component in _extract_components(foreground):
        component_mask = component["mask"]
        area = int(component["area"])
        if area < 18:
            continue
        component_mean = roi_preview[component_mask].mean(axis=0)
        mean_r, mean_g, mean_b = [float(value) for value in component_mean]
        component_intensity = float(mean_intensity[component_mask].mean())
        component_spread = float(spread[component_mask].mean())
        component_distance = float(background_distance[component_mask].mean())
        component_band_std = float(band_std[component_mask].mean()) if band_std.size else 0.0
        fill_ratio = float(component["fill_ratio"])
        elongatedness = float(component["elongatedness"])
        boundary_density = float(component["boundary_pixels"]) / max(area, 1.0)
        green_dominant = mean_g >= mean_r + 6.0 and mean_g >= mean_b + 4.0
        yellowish = mean_r >= 120.0 and mean_g >= 110.0 and abs(mean_r - mean_g) <= 45.0
        thin_linear = elongatedness >= 2.8 or (min(component["bbox_width"], component["bbox_height"]) <= 10 and elongatedness >= 1.8)
        irregular = boundary_density >= 0.22 or fill_ratio <= 0.52
        compact_paint = fill_ratio >= 0.42 and boundary_density <= 0.18
        detached_small = area <= int(roi_area * 0.22)

        if component_intensity >= bright_threshold and component_spread <= low_chroma_threshold and thin_linear:
            roi_masks["aging"] |= component_mask
            continue

        if area >= 36 and detached_small and (green_dominant or yellowish) and (irregular or component_distance >= deviation_threshold * 1.05):
            roi_masks["pollution"] |= component_mask
            continue

        if area >= 48 and irregular and detached_small and component_distance >= deviation_threshold * 0.95 and component_band_std <= max(spectral_threshold * 1.4, 0.03):
            roi_masks["pollution"] |= component_mask
            continue

        if area >= 36 and (component_spread >= spread_threshold * 0.9 or component_band_std >= spectral_threshold * 0.9 or compact_paint):
            roi_masks["paint"] |= component_mask
            continue

        if area >= 64 and not component["touches_border"]:
            roi_masks["paint"] |= component_mask

    residual_aging = (mean_intensity >= bright_threshold) & (spread <= low_chroma_threshold) & ~(roi_masks["paint"] | roi_masks["pollution"] | roi_masks["aging"])
    for component in _extract_components(_morphology_cleanup(residual_aging, close_iterations=1, open_iterations=0)):
        if int(component["area"]) < 12:
            continue
        if float(component["elongatedness"]) >= 2.0 or min(component["bbox_width"], component["bbox_height"]) <= 8:
            roi_masks["aging"] |= component["mask"]

    output_masks = {head_name: np.zeros((height, width), dtype=np.uint8) for head_name in roi_masks}
    for head_name, roi_mask in roi_masks.items():
        cleaned = _fill_mask_holes(_morphology_cleanup(roi_mask, close_iterations=1, open_iterations=0))
        final_mask = np.zeros((height, width), dtype=np.uint8)
        final_mask[roi_top:roi_bottom, roi_left:roi_right] = cleaned.astype(np.uint8) * 255
        output_masks[head_name] = final_mask
    return output_masks


def _seed_scene_annotations_from_rules(scene_id: str, scenes_root: Path = SCENES_ROOT) -> dict[str, Any]:
    scene_root = scenes_root / scene_id
    preview_array = np.asarray(Image.open(scene_root / "preview.png").convert("RGB"), dtype=np.uint8)
    five_band_path = scene_root / "five_band.npy"
    five_band = np.load(five_band_path).astype(np.float32) if five_band_path.exists() else None
    try:
        roi_bbox = detect_board_bbox_from_preview(preview_array)
    except ValueError:
        roi_bbox = (0, 0, preview_array.shape[1], preview_array.shape[0])
    masks = _build_rule_based_seed_masks(preview_array, five_band=five_band, roi_bbox=roi_bbox)
    _write_seed_masks(scene_root, masks)
    overlay_path = _render_annotation_overlay(scene_id, scenes_root=scenes_root)
    return {
        "scene_id": scene_id,
        "roi_bbox": [int(value) for value in roi_bbox],
        "annotation_overlay": str(overlay_path),
        "positive_pixels": {head_name: int((mask > 0).sum()) for head_name, mask in masks.items()},
    }



def _component_boundary(mask: np.ndarray) -> np.ndarray:
    up = np.roll(mask, -1, axis=0)
    down = np.roll(mask, 1, axis=0)
    left = np.roll(mask, -1, axis=1)
    right = np.roll(mask, 1, axis=1)
    up[-1, :] = False
    down[0, :] = False
    left[:, -1] = False
    right[:, 0] = False
    interior = up & down & left & right
    return mask & (~interior)


def _extract_components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, Any]] = []
    starts = np.argwhere(mask)
    for start_y, start_x in starts:
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        min_y = max_y = int(start_y)
        min_x = max_x = int(start_x)
        touches_border = False
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            if y == 0 or x == 0 or y == height - 1 or x == width - 1:
                touches_border = True
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        component_mask = np.zeros_like(mask, dtype=bool)
        ys = np.fromiter((item[0] for item in pixels), dtype=np.int32)
        xs = np.fromiter((item[1] for item in pixels), dtype=np.int32)
        component_mask[ys, xs] = True
        bbox_height = max_y - min_y + 1
        bbox_width = max_x - min_x + 1
        bbox_area = bbox_height * bbox_width
        area = int(len(pixels))
        fill_ratio = area / float(bbox_area) if bbox_area else 0.0
        boundary_pixels = int(np.count_nonzero(_component_boundary(component_mask)))
        elongatedness = max(bbox_height, bbox_width) / max(1.0, float(min(bbox_height, bbox_width)))
        components.append(
            {
                "mask": component_mask,
                "area": area,
                "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
                "bbox_width": int(bbox_width),
                "bbox_height": int(bbox_height),
                "fill_ratio": float(fill_ratio),
                "touches_border": touches_border,
                "boundary_pixels": boundary_pixels,
                "elongatedness": float(elongatedness),
            }
        )
    return components


def _keep_component_for_head(head_name: str, candidate_kind: str, component: dict[str, Any]) -> bool:
    area = int(component["area"])
    fill_ratio = float(component["fill_ratio"])
    elongatedness = float(component["elongatedness"])
    touches_border = bool(component["touches_border"])
    bbox_width = int(component["bbox_width"])
    bbox_height = int(component["bbox_height"])

    if head_name == "paint":
        if area < 96 or touches_border:
            return False
        if candidate_kind == "missing":
            return fill_ratio >= 0.10 and max(bbox_width, bbox_height) >= 18
        return area >= 64 and fill_ratio >= 0.06

    if head_name == "pollution":
        if area < 48 or touches_border:
            return False
        if candidate_kind == "missing":
            return fill_ratio <= 0.82 or max(bbox_width, bbox_height) >= 14
        return fill_ratio <= 0.90 or area >= 96

    if head_name == "aging":
        if area < 8:
            return False
        if candidate_kind == "missing":
            return elongatedness >= 1.8 or fill_ratio <= 0.55 or component["boundary_pixels"] >= 12
        return elongatedness >= 1.4 or fill_ratio <= 0.70

    return area > 0


def _refine_audit_diff(head_name: str, candidate_kind: str, diff_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    refined = np.zeros_like(diff_mask, dtype=bool)
    kept_components: list[dict[str, Any]] = []
    for component in _extract_components(diff_mask):
        if not _keep_component_for_head(head_name, candidate_kind, component):
            continue
        refined |= component["mask"]
        kept_components.append(component)
    return refined, kept_components


def _write_audit_overlay(preview: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], output_path: Path) -> None:
    Image.fromarray(make_overlay_image(preview, mask.astype(np.uint8) * 255, color)).save(output_path)


def _generate_annotation_audit(*, scene_id: str, prediction_root: Path, adjust_target: str, scenes_root: Path = SCENES_ROOT) -> dict[str, Any]:
    scene_root = scenes_root / scene_id
    sample_prediction_root = prediction_root / scene_id
    audit_root = sample_prediction_root / AUDIT_ROOT_NAME
    if audit_root.exists():
        shutil.rmtree(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)

    preview = np.asarray(Image.open(scene_root / "preview.png").convert("RGB"), dtype=np.uint8)
    selected_heads = ("paint",) if adjust_target == "paint" else ("pollution", "aging")
    summary: dict[str, Any] = {
        "available": True,
        "target": adjust_target,
        "audit_root": str(audit_root),
        "heads": {},
    }
    total_flags = 0

    for head_name in selected_heads:
        label_mask = _load_binary_mask(scene_root / "masks" / f"{head_name}.png")
        pred_mask = _load_binary_mask(sample_prediction_root / f"{head_name}_pred.png")
        missing_mask, missing_components = _refine_audit_diff(head_name, "missing", pred_mask & (~label_mask))
        overmark_mask, overmark_components = _refine_audit_diff(head_name, "overmark", label_mask & (~pred_mask))
        total_flags += int(np.count_nonzero(missing_mask)) + int(np.count_nonzero(overmark_mask))

        head_summary: dict[str, Any] = {}
        for candidate_kind, mask, components in (
            ("missing", missing_mask, missing_components),
            ("overmark", overmark_mask, overmark_components),
        ):
            mask_name = f"{head_name}_{candidate_kind}_mask.png"
            overlay_name = f"{head_name}_{candidate_kind}_overlay.png"
            _save_binary_mask(audit_root / mask_name, mask)
            _write_audit_overlay(preview, mask, AUDIT_DIFF_COLORS[candidate_kind], audit_root / overlay_name)
            head_summary[candidate_kind] = {
                "positive_pixels": int(np.count_nonzero(mask)),
                "component_count": len(components),
                "mask": mask_name,
                "overlay": overlay_name,
                "review_hint": "missing" if candidate_kind == "missing" else "overmark",
            }
        summary["heads"][head_name] = head_summary

    summary["flagged_pixels"] = total_flags
    (audit_root / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _scene_numeric_id(scene_id: str) -> int | None:
    if not scene_id.startswith("SAMPLE_"):
        return None
    suffix = scene_id.split("_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def _should_use_paint_override(scene_id: str) -> bool:
    numeric_id = _scene_numeric_id(scene_id)
    return numeric_id is not None and numeric_id >= PAINT_OVERRIDE_SCENE_START


def _generate_review_predictions(scene_id: str) -> Path:
    export_five_band_predictions(
        checkpoint_path=VALIDATION_CHECKPOINT_PATH,
        scenes_root=SCENES_ROOT,
        output_root=MAIN_PREDICTION_ROOT,
        scene_ids=[scene_id],
        composition_mode="conflict_resolved",
        pollution_shape_filter=True,
        pollution_threshold=0.35,
    )
    if _should_use_paint_override(scene_id):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        paint_override_root = TMP_ROOT / scene_id / "paint_override"
        if paint_override_root.exists():
            shutil.rmtree(paint_override_root)
        export_five_band_predictions(
            checkpoint_path=PAINT_OVERRIDE_CHECKPOINT_PATH,
            scenes_root=SCENES_ROOT,
            output_root=paint_override_root,
            scene_ids=[scene_id],
            export_heads=("paint",),
            composition_mode="independent",
        )
        compose_predictions(
            paint_root=paint_override_root,
            pollution_root=MAIN_PREDICTION_ROOT,
            aging_root=MAIN_PREDICTION_ROOT,
            scenes_root=SCENES_ROOT,
            output_root=MAIN_PREDICTION_ROOT,
            scene_ids=[scene_id],
        )
    return MAIN_PREDICTION_ROOT


def _state_response(state: dict[str, Any]) -> dict[str, Any]:
    records = []
    for scene_id in sorted(state.get("samples", {})):
        record = dict(state["samples"][scene_id])
        record["scene_id"] = scene_id
        record["status_label"] = STATUS_LABELS.get(str(record.get("status")), str(record.get("status") or "未知"))
        records.append(record)
    return {
        "samples": records,
        "updated_at": state.get("updated_at"),
        "count": len(records),
    }


def _import_directory(state: dict[str, Any], source_dir: Path, *, background_role: str, direction: str, light_level: str, role: str = "mother", parent_scene_id: str | None = None) -> tuple[str, bool]:
    source_directory = str(source_dir.resolve())
    for scene_id, record in state.get("samples", {}).items():
        if record.get("source_directory") == source_directory:
            return scene_id, True
    scene_id = next_sample_id()
    export_camera_scene_from_images(scene_id, select_latest_camera_band_images(source_dir), SCENES_ROOT)
    _write_scene_metadata(scene_id, background_role=background_role, light_level=light_level, direction=direction, source_directory=source_directory)
    state.setdefault("samples", {})[scene_id] = {
        "status": "awaiting_annotation", "stage": "awaiting_annotation", "display_prediction_mode": "none",
        "source_directory": source_directory, "background_role": background_role, "light_level": light_level,
        "direction": direction, "role": role, "parent_scene_id": parent_scene_id, "updated_at": _now_iso(),
        "annotation_source": "blank_manual", "annotation_source_label": "空白标注层（尚未预测）",
        "annotation_decision": "pending",
    }
    return scene_id, False


def import_validation_capture_group(
    image_paths: Sequence[Path],
    background_role: str,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """Import one exact five-image capture group from the camera root.

    A camera root can contain several consecutive captures, so its directory
    alone must never be used as the de-duplication key for validation data.
    """
    resolved_paths = [Path(path).resolve() for path in image_paths]
    if len(resolved_paths) != 5 or any(not path.exists() for path in resolved_paths):
        raise ValueError("验证集导入必须提供同一采集组的 5 个存在的 BMP 文件。")
    if len({path.parent for path in resolved_paths}) != 1:
        raise ValueError("验证集的 5 个波段文件必须来自同一目录。")
    source_key = "capture_group:" + "|".join(path.name for path in resolved_paths)
    state = load_workflow_state(state_path)
    for scene_id, record in state.get("samples", {}).items():
        if record.get("source_group_key") == source_key:
            return {"action": "import_validation_capture", "scene_id": scene_id, "reused": True, "record": record}
    scene_id = next_sample_id()
    export_camera_scene_from_images(scene_id, resolved_paths, SCENES_ROOT)
    _write_scene_metadata(scene_id, background_role=background_role, light_level="unknown", direction="unknown", source_directory=str(resolved_paths[0].parent))
    record = {
        "status": "awaiting_annotation", "stage": "awaiting_annotation", "display_prediction_mode": "none",
        "source_directory": str(resolved_paths[0].parent), "source_group_key": source_key,
        "source_images": [str(path) for path in resolved_paths], "background_role": background_role,
        "light_level": "unknown", "direction": "unknown", "role": "validation", "parent_scene_id": None,
        "updated_at": _now_iso(), "annotation_source": "blank_manual",
        "annotation_source_label": "验证样本：未进入训练，等待 background4_v1 预测",
        "annotation_decision": "pending",
    }
    state.setdefault("samples", {})[scene_id] = record
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "import_validation_capture", "scene_id": scene_id, "reused": False, "record": record, "manifest_sample_count": len(manifest.get("samples", []))}


def import_latest_capture(state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    image_paths = discover_latest_capture_group()
    source_dir = image_paths[0].parent
    scene_id, reused = _import_directory(state, source_dir, background_role="代赭", direction="unknown", light_level="5")
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "import_latest", "scene_id": scene_id, "reused": reused, "record": state["samples"][scene_id], "manifest_sample_count": len(manifest.get("samples", []))}


def import_background4_mothers(state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    results = []
    for relative, background, direction, light, role in BACKGROUND4_IMPORTS:
        source_dir = DATA_ROOT / relative
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        scene_id, reused = _import_directory(state, source_dir, background_role=background, direction=direction, light_level=light, role=role)
        results.append({"scene_id": scene_id, "source_directory": str(source_dir), "reused": reused})
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "import_background4_mothers", "samples": results, "manifest_sample_count": len(manifest.get("samples", []))}


def import_background4_pure_backgrounds(state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Import the stone-blue/stone-green blank boards as background negatives."""
    state = load_workflow_state(state_path)
    results = []
    for relative, background, light in BACKGROUND4_PURE_BACKGROUNDS:
        source_dir = DATA_ROOT / relative
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        scene_id, reused = _import_directory(
            state, source_dir, background_role=background, direction="background",
            light_level=light, role="pure_background",
        )
        results.append({"scene_id": scene_id, "source_directory": str(source_dir), "reused": reused})
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "import_background4_pure_backgrounds", "samples": results, "manifest_sample_count": len(manifest.get("samples", []))}


def run_review_seed(scene_id: str, reset: bool = False, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    record = state.get("samples", {}).get(scene_id)
    if record is None:
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    if record.get("annotation_decision") in {"saved", "accepted"} and not reset:
        raise ValueError("Saved annotations are protected; use the explicit reset action to replace them with a prediction draft.")
    root = _generate_review_predictions(scene_id)
    _seed_scene_annotations_from_prediction(scene_id, root)
    record.update({"status": "pending_review", "stage": "review", "prediction_root": str(root), "display_prediction_mode": "validation", **_default_annotation_fields(), "updated_at": _now_iso()})
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "run_review_seed", "scene_id": scene_id, "record": record, "manifest_sample_count": len(manifest.get("samples", []))}


def run_background4_prediction(scene_id: str, version_id: str = "background4_v2", state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Run the trained dense-pigment model without touching saved annotations."""
    state = load_workflow_state(state_path)
    record = state.get("samples", {}).get(scene_id)
    if record is None:
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    if not record.get("background_role"):
        raise ValueError(f"请先选择并保存背景板，再运行 {version_id} 预测。")
    versions = {
        "background4_v1": (BACKGROUND4_CHECKPOINT_PATH, BACKGROUND4_PREDICTION_ROOT),
        "background4_v2": (BACKGROUND4_V2_CHECKPOINT_PATH, BACKGROUND4_V2_PREDICTION_ROOT),
        "background4_v3": (BACKGROUND4_V3_CHECKPOINT_PATH, BACKGROUND4_V3_PREDICTION_ROOT),
        "background4_v3_agingfix_v1_best": (
            BACKGROUND4_V3_AGINGFIX_CHECKPOINT_PATH,
            BACKGROUND4_V3_AGINGFIX_PREDICTION_ROOT,
        ),
    }
    if version_id not in versions:
        raise ValueError(f"Unsupported background4 version: {version_id}")
    checkpoint_path, prediction_root = versions[version_id]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{version_id} 尚未完成正式训练：{checkpoint_path}")
    exported = export_five_band_predictions(
        checkpoint_path=checkpoint_path,
        scenes_root=SCENES_ROOT,
        output_root=prediction_root,
        scene_ids=[scene_id],
        threshold=0.5,
        background_role=str(record["background_role"]),
    )
    # A new background4 prediction supplies all three binary heads plus the
    # dense four-class pigment draft.  Saved/accepted work remains protected.
    if record.get("annotation_decision") not in {"saved", "accepted"}:
        _seed_scene_annotations_from_prediction(scene_id, prediction_root)
    record.update({
        "prediction_root": str(prediction_root),
        "pigment_root": str(prediction_root),
        "display_prediction_mode": "workflow",
        "background4_prediction": True,
        "prediction_version_id": version_id,
        "updated_at": _now_iso(),
    })
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "run_background4_prediction", "scene_id": scene_id, "exported_count": len(exported), "record": record, "manifest_sample_count": len(manifest.get("samples", []))}


def _create_light_derivatives(state: dict[str, Any], scene_id: str) -> list[str]:
    """Create 1/10-light review copies after the 5-light mother was manually saved."""
    parent = state["samples"][scene_id]
    if parent.get("role") != "mother" or str(parent.get("light_level")) != "5":
        return []
    source_dir = Path(str(parent["source_directory"]))
    created: list[str] = []
    for light in ("1", "10"):
        candidate = source_dir.parent / light
        if not candidate.exists():
            continue
        child_id, reused = _import_directory(state, candidate, background_role=str(parent["background_role"]), direction=str(parent.get("direction") or "A"), light_level=light, role="light_derivative", parent_scene_id=scene_id)
        if not reused:
            src_masks = SCENES_ROOT / scene_id / "masks"
            dst_masks = SCENES_ROOT / child_id / "masks"
            for name in ("paint.png", "pollution.png", "aging.png", "pigment.png"):
                source = src_masks / name
                if source.exists(): shutil.copy2(source, dst_masks / name)
            state["samples"][child_id].update({"status": "awaiting_annotation", "stage": "light_transfer_review", "annotation_source": "transferred_from_light5", "annotation_source_label": f"由 {scene_id} 光照5标注复制，需人工复核并保存", "annotation_decision": "pending"})
            created.append(child_id)
    return created


def propagate_annotation_to_dataset(scene_id: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Copy a saved mother annotation to matching dataN direction/light captures.

    A/a is the reference.  B is a clockwise 90-degree capture, rotated in the
    original landscape canvas without aspect-ratio stretching; C is a
    180-degree capture.  C also receives a conservative image-based
    translation correction for camera framing drift.  All copies are
    explicitly marked for human review and are never auto-approved.
    """
    state = load_workflow_state(state_path)
    source_record = state.get("samples", {}).get(scene_id)
    if source_record is None or source_record.get("annotation_source") != "manual_from_review":
        raise ValueError("Propagation requires a manually saved source annotation.")
    source_directory = Path(str(source_record["source_directory"]))
    dataset_root = source_directory.parents[1]
    source_direction = source_directory.parent.name.lower()
    if source_direction not in {"a", "b", "c"}:
        raise ValueError("Propagation is available only for A/B/C acquisition directories.")
    source_masks = SCENES_ROOT / scene_id / "masks"
    targets: list[dict[str, str]] = []
    # np.rot90 uses counter-clockwise turns.  The acquisition's B direction
    # is clockwise relative to A, not counter-clockwise.
    turns_by_direction = {"a": 0, "b": -1, "c": 2}
    for direction in ("a", "b", "c"):
        for light in ("1", "5", "10"):
            directory = dataset_root / direction / light
            if directory.resolve() == source_directory.resolve() or not directory.exists():
                continue
            target_id, _ = _import_directory(state, directory, background_role=str(source_record["background_role"]), direction=direction.upper(), light_level=light, role="annotation_transfer", parent_scene_id=scene_id)
            target_record = state["samples"][target_id]
            if target_record.get("annotation_decision") == "accepted":
                continue
            prior_transform = dict(target_record.get("annotation_transform") or {})
            prior_manual_shift = prior_transform.get("manual_shift_yx", [0, 0])
            if not isinstance(prior_manual_shift, (list, tuple)) or len(prior_manual_shift) != 2:
                prior_manual_shift = [0, 0]
            manual_shift = (int(prior_manual_shift[0]), int(prior_manual_shift[1]))
            target_shape = _load_scene_preview_shape(target_id)
            relative_turns = (turns_by_direction[direction] - turns_by_direction[source_direction]) % 4
            target_masks = SCENES_ROOT / target_id / "masks"
            c_shift = (0, 0)
            if direction == "c":
                source_preview = np.asarray(Image.open(SCENES_ROOT / scene_id / "preview.png").convert("L"), dtype=np.uint8)
                target_preview = np.asarray(Image.open(SCENES_ROOT / target_id / "preview.png").convert("L"), dtype=np.uint8)
                if relative_turns % 2:
                    rotation_angle = 90 if relative_turns == 1 else -90
                    rotated_preview = ndimage.rotate(source_preview, angle=rotation_angle, reshape=False, order=1, mode="constant", cval=0, prefilter=False).astype(np.uint8)
                else:
                    rotated_preview = np.rot90(source_preview, relative_turns) if relative_turns else source_preview
                if rotated_preview.shape != target_shape:
                    rotated_preview = np.asarray(Image.fromarray(rotated_preview).resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR), dtype=np.uint8)
                # Use low-frequency-normalized phase correlation only for a
                # small framing correction.  Larger values indicate that the
                # images are not safely comparable and are intentionally ignored.
                source_small = ndimage.zoom(rotated_preview.astype(np.float32), (min(360, target_shape[0]) / target_shape[0], min(576, target_shape[1]) / target_shape[1]), order=1)
                target_small = ndimage.zoom(target_preview.astype(np.float32), (source_small.shape[0] / target_shape[0], source_small.shape[1] / target_shape[1]), order=1)
                source_edges = ndimage.sobel(source_small, axis=0) + ndimage.sobel(source_small, axis=1)
                target_edges = ndimage.sobel(target_small, axis=0) + ndimage.sobel(target_small, axis=1)
                spectrum = np.fft.fft2(target_edges) * np.conj(np.fft.fft2(source_edges))
                spectrum /= np.maximum(np.abs(spectrum), 1e-8)
                peak_y, peak_x = np.unravel_index(np.argmax(np.abs(np.fft.ifft2(spectrum))), source_edges.shape)
                shift_y = peak_y if peak_y <= source_edges.shape[0] // 2 else peak_y - source_edges.shape[0]
                shift_x = peak_x if peak_x <= source_edges.shape[1] // 2 else peak_x - source_edges.shape[1]
                candidate = (round(shift_y * target_shape[0] / source_edges.shape[0]), round(shift_x * target_shape[1] / source_edges.shape[1]))
                if max(abs(candidate[0]), abs(candidate[1])) <= 48:
                    c_shift = candidate
            for name in ("paint", "pollution", "aging", "pigment"):
                source_path = source_masks / f"{name}.png"
                if not source_path.exists():
                    continue
                array = np.asarray(Image.open(source_path).convert("L"), dtype=np.uint8)
                if relative_turns % 2:
                    # A 90-degree rotation swaps width and height.  Do not
                    # resize that portrait result back into a landscape frame:
                    # it would stretch every annotated region.  Rotate around
                    # the same landscape canvas instead.
                    rotation_angle = 90 if relative_turns == 1 else -90
                    array = ndimage.rotate(array, angle=rotation_angle, reshape=False, order=0, mode="constant", cval=0, prefilter=False).astype(np.uint8)
                elif relative_turns:
                    array = np.rot90(array, relative_turns)
                if array.shape != target_shape:
                    array = np.asarray(Image.fromarray(array).resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
                total_shift = (c_shift[0] + manual_shift[0], c_shift[1] + manual_shift[1])
                if total_shift != (0, 0):
                    array = ndimage.shift(array, shift=total_shift, order=0, mode="constant", cval=0, prefilter=False).astype(np.uint8)
                Image.fromarray(array).save(target_masks / f"{name}.png")
            transform = {"rotation_degrees": int(relative_turns * 90), "translation_yx": list(c_shift)}
            if manual_shift != (0, 0):
                transform["manual_shift_yx"] = list(manual_shift)
            target_record.update({"status": "pending_review", "stage": "transferred_review", "annotation_source": "transferred_from_manual", "annotation_source_label": f"由 {scene_id} 自动迁移，需人工复核", "annotation_transform": transform, "annotation_decision": "pending", "updated_at": _now_iso()})
            targets.append({"scene_id": target_id, "source_directory": str(directory)})
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "propagate_annotation", "source_scene_id": scene_id, "target_count": len(targets), "targets": targets, "manifest_sample_count": len(manifest.get("samples", []))}


def rerun_review(scene_id: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    if scene_id not in state.get("samples", {}):
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    prediction_root = _generate_review_predictions(scene_id)
    record = state["samples"][scene_id]
    record["status"] = "pending_review"
    record["stage"] = "review"
    record["prediction_root"] = str(prediction_root)
    record["display_prediction_mode"] = "validation"
    record.update(_default_annotation_fields())
    record.pop("annotation_audit", None)
    record["updated_at"] = _now_iso()
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {
        "action": "rerun_review",
        "scene_id": scene_id,
        "record": record,
        "manifest_sample_count": len(manifest.get("samples", [])),
    }


def save_annotation(scene_id: str, payload: dict[str, Any], state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    if scene_id not in state.get("samples", {}):
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    scene_root = SCENES_ROOT / scene_id
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene workspace not found: {scene_root}")

    expected_shape = _load_scene_preview_shape(scene_id, SCENES_ROOT)
    masks_payload = None
    if isinstance(payload, dict):
        nested_masks = payload.get("masks")
        if isinstance(nested_masks, dict):
            masks_payload = nested_masks
        elif all(head_name in payload for head_name in ("paint", "pollution", "aging")):
            masks_payload = payload
    if not isinstance(masks_payload, dict):
        raise ValueError("Expected JSON payload with masks for paint / pollution / aging.")

    decoded_masks: dict[str, np.ndarray] = {}
    for head_name in ("paint", "pollution", "aging"):
        encoded = masks_payload.get(head_name)
        if encoded is None:
            raise ValueError(f"Missing mask payload for {head_name}.")
        mask = _decode_mask_payload(str(encoded))
        _validate_mask_shape(scene_id, mask, expected_shape)
        decoded_masks[head_name] = mask

    _write_seed_masks(scene_root, decoded_masks)
    pigment_encoded = masks_payload.get("pigment")
    if pigment_encoded is not None:
        pigment = _decode_label_payload(str(pigment_encoded))
        _validate_mask_shape(scene_id, pigment, expected_shape)
        pigment = pigment.copy()
        pigment[(pigment > 4) | (~decoded_masks["paint"])] = 0
        Image.fromarray(pigment.astype(np.uint8)).save(scene_root / "masks" / "pigment.png")

    record = state["samples"][scene_id]
    record["status"] = "pending_review"
    record["stage"] = "annotation_saved"
    record["annotation_source"] = "manual_from_review"
    record["annotation_source_label"] = "人工修订 review 草稿"
    record["annotation_rule_source"] = _annotation_rule_source()
    record["annotation_rule_label"] = "人工已修订并保存 masks；当前作为最终标注保留"
    record["annotation_decision"] = "saved"
    record.pop("annotation_audit", None)
    record["updated_at"] = _now_iso()
    derived_scene_ids = _create_light_derivatives(state, scene_id)

    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {
        "action": "save_annotation",
        "scene_id": scene_id,
        "record": record,
        "derived_scene_ids": derived_scene_ids,
        "manifest_sample_count": len(manifest.get("samples", [])),
    }


def hold_scene(scene_id: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    if scene_id not in state.get("samples", {}):
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    record = state["samples"][scene_id]
    record["status"] = "held"
    record["stage"] = "review"
    record["updated_at"] = _now_iso()
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {
        "action": "hold_scene",
        "scene_id": scene_id,
        "record": record,
        "manifest_sample_count": len(manifest.get("samples", [])),
    }


def approve_scene(scene_id: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    if scene_id not in state.get("samples", {}):
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    record = state["samples"][scene_id]
    scene_root = SCENES_ROOT / scene_id
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root not found: {scene_root}")

    record.update(
        {
            "status": "approved",
            "stage": "annotation_approved",
            "display_prediction_mode": "validation",
            "updated_at": _now_iso(),
            "annotation_decision": "accepted",
            "adjust_target": None,
        }
    )
    if record.get("annotation_source") != "manual_from_review":
        raise ValueError("Only manually saved annotations can be adopted into background4_v1 patches.")
    patch_result = export_patch_dataset(scene_root, "train", BACKGROUND4_PATCH_ROOT, patch_size=256, stride=192)
    record["background4_patch_root"] = str(BACKGROUND4_PATCH_ROOT)
    record["background4_patch_count"] = patch_result["patches"]
    record.pop("patch_root", None)
    record.pop("train_dir", None)
    record.pop("last_train_stdout", None)
    record.pop("last_predict_stdout", None)
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {
        "action": "approve_scene",
        "scene_id": scene_id,
        "record": record,
        "manifest_sample_count": len(manifest.get("samples", [])),
    }


def batch_approve_reviewed_scenes(scene_ids: Sequence[str], state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Adopt a user-confirmed batch without changing historical softcomp data."""
    state = load_workflow_state(state_path)
    results: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        record = state.get("samples", {}).get(scene_id)
        if record is None:
            raise KeyError(f"Unknown workflow scene: {scene_id}")
        if record.get("annotation_decision") == "accepted":
            continue
        scene_root = SCENES_ROOT / scene_id
        if not scene_root.exists():
            raise FileNotFoundError(scene_root)
        is_pure_background = record.get("role") == "pure_background" or (record.get("direction") == "background" and record.get("annotation_source") == "blank_manual")
        if is_pure_background:
            shape = _load_scene_preview_shape(scene_id)
            _write_seed_masks(scene_root, {name: np.zeros(shape, dtype=bool) for name in ("paint", "pollution", "aging")})
            Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(scene_root / "masks" / "pigment.png")
            record["annotation_source_label"] = "人工确认纯背景：三头与颜料标签均为零"
        record.update({
            "status": "approved", "stage": "annotation_approved", "display_prediction_mode": "validation",
            "updated_at": _now_iso(), "annotation_decision": "accepted", "annotation_source": "manual_from_review",
            "adjust_target": None,
        })
        patch_result = export_patch_dataset(scene_root, "train", BACKGROUND4_PATCH_ROOT, patch_size=256, stride=192)
        record["background4_patch_root"] = str(BACKGROUND4_PATCH_ROOT)
        record["background4_patch_count"] = patch_result["patches"]
        results.append({"scene_id": scene_id, "patches": patch_result["patches"], "pure_background": is_pure_background})
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "batch_approve_reviewed_scenes", "results": results, "manifest_sample_count": len(manifest.get("samples", []))}


def approve_background4_v3_scenes(
    scene_ids: Sequence[str],
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """Approve reviewed v3 annotations and export only to the isolated v3 root."""
    state = load_workflow_state(state_path)
    results: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        record = state.get("samples", {}).get(scene_id)
        if record is None:
            raise KeyError(f"Unknown workflow scene: {scene_id}")
        if record.get("annotation_source") not in {"manual_from_review", "transferred_from_manual"}:
            raise ValueError(f"{scene_id} has no reviewed annotation to export.")
        scene_root = SCENES_ROOT / scene_id
        shape = _load_scene_preview_shape(scene_id)
        masks_root = scene_root / "masks"
        masks = {
            name: np.asarray(Image.open(masks_root / f"{name}.png").convert("L"), dtype=np.uint8)
            for name in ("paint", "pollution", "aging", "pigment")
        }
        if any(mask.shape != shape for mask in masks.values()):
            raise ValueError(f"{scene_id} mask shape does not match its preview.")
        if record.get("role") == "pure_aging_candidate":
            if np.any(masks["paint"] > 0) or np.any(masks["pollution"] > 0) or np.any(masks["pigment"] > 0):
                raise ValueError(f"{scene_id} pure-aging sample must keep paint/pollution/pigment empty.")
            if not np.any(masks["aging"] > 0):
                raise ValueError(f"{scene_id} pure-aging sample has no aging pixels.")
        if np.any((masks["pigment"] > 0) & ~(masks["paint"] > 0)):
            raise ValueError(f"{scene_id} has pigment outside paint.")
        patch_result = export_patch_dataset(
            scene_root,
            "train",
            BACKGROUND4_V3_PATCH_ROOT,
            patch_size=256,
            stride=192,
        )
        record.update(
            {
                "status": "approved",
                "stage": "annotation_approved",
                "display_prediction_mode": "none",
                "annotation_decision": "accepted",
                "approval_basis": "user_confirmed_background4_v3_2026-07-23",
                "background4_v3_patch_root": str(BACKGROUND4_V3_PATCH_ROOT),
                "background4_v3_patch_count": patch_result["patches"],
                "updated_at": _now_iso(),
            }
        )
        results.append({"scene_id": scene_id, "patches": patch_result["patches"]})
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {
        "action": "approve_background4_v3_scenes",
        "results": results,
        "manifest_sample_count": len(manifest.get("samples", [])),
    }


def delete_scene(scene_id: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Remove an unapproved test scene and its unapproved light derivatives."""
    state = load_workflow_state(state_path)
    record = state.get("samples", {}).get(scene_id)
    if record is None:
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    if record.get("annotation_decision") == "accepted":
        raise ValueError("Approved scenes cannot be deleted through the UI workflow.")
    targets = [scene_id] + [key for key, value in state["samples"].items() if value.get("parent_scene_id") == scene_id and value.get("annotation_decision") != "accepted"]
    for target in targets:
        scene_root = SCENES_ROOT / target
        if scene_root.exists():
            shutil.rmtree(scene_root)
        state["samples"].pop(target, None)
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "delete_scene", "deleted_scene_ids": targets, "manifest_sample_count": len(manifest.get("samples", []))}


def renumber_scene(scene_id: str, target_scene_id: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    if scene_id not in state.get("samples", {}):
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    if target_scene_id in state["samples"] or (SCENES_ROOT / target_scene_id).exists():
        raise ValueError(f"Target scene already exists: {target_scene_id}")
    source_root = SCENES_ROOT / scene_id
    target_root = SCENES_ROOT / target_scene_id
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    shutil.move(str(source_root), str(target_root))
    record = state["samples"].pop(scene_id)
    record["updated_at"] = _now_iso()
    state["samples"][target_scene_id] = record
    for value in state["samples"].values():
        if value.get("parent_scene_id") == scene_id:
            value["parent_scene_id"] = target_scene_id
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "renumber_scene", "scene_id": target_scene_id, "previous_scene_id": scene_id, "manifest_sample_count": len(manifest.get("samples", []))}


def clear_scene_annotation(scene_id: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    if scene_id not in state.get("samples", {}):
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    shape = _load_scene_preview_shape(scene_id)
    scene_root = SCENES_ROOT / scene_id
    _write_seed_masks(scene_root, {name: np.zeros(shape, dtype=bool) for name in ("paint", "pollution", "aging")})
    pigment_path = scene_root / "masks" / "pigment.png"
    if pigment_path.exists():
        pigment_path.unlink()
    record = state["samples"][scene_id]
    record.update({"status": "awaiting_annotation", "stage": "awaiting_annotation", "display_prediction_mode": "none", "annotation_source": "blank_manual", "annotation_source_label": "空白标注层（测试标注已清空）", "annotation_decision": "pending", "updated_at": _now_iso()})
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "clear_scene_annotation", "scene_id": scene_id, "manifest_sample_count": len(manifest.get("samples", []))}


def shift_annotations(scene_ids: Sequence[str], shift_y: int, shift_x: int = 0, state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Apply one reviewed geometric correction to all annotation layers."""
    state = load_workflow_state(state_path)
    changed: list[str] = []
    for scene_id in scene_ids:
        record = state.get("samples", {}).get(scene_id)
        if record is None:
            raise KeyError(f"Unknown workflow scene: {scene_id}")
        if record.get("annotation_decision") == "accepted":
            raise ValueError(f"Refusing to shift an accepted training annotation: {scene_id}")
        masks_root = SCENES_ROOT / scene_id / "masks"
        for name in ("paint", "pollution", "aging", "pigment"):
            path = masks_root / f"{name}.png"
            if not path.exists():
                continue
            array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            shifted = ndimage.shift(array, shift=(shift_y, shift_x), order=0, mode="constant", cval=0, prefilter=False).astype(np.uint8)
            Image.fromarray(shifted).save(path)
        previous = dict(record.get("annotation_transform") or {})
        previous["manual_shift_yx"] = [int(shift_y), int(shift_x)]
        record.update({"annotation_transform": previous, "annotation_source_label": f"由 {record.get('parent_scene_id') or '母本'} 自动迁移，已人工确认上移校正，需复核", "updated_at": _now_iso()})
        changed.append(scene_id)
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "shift_annotations", "scene_ids": changed, "shift_yx": [int(shift_y), int(shift_x)], "manifest_sample_count": len(manifest.get("samples", []))}


def set_background_role(scene_id: str, background_role: str, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    record = state.get("samples", {}).get(scene_id)
    if record is None:
        raise KeyError(f"Unknown workflow scene: {scene_id}")
    if background_role not in BACKGROUND_ROLES:
        raise ValueError(f"Unsupported background role: {background_role}")
    record["background_role"] = background_role
    record["updated_at"] = _now_iso()
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {"action": "set_background_role", "scene_id": scene_id, "background_role": background_role, "manifest_sample_count": len(manifest.get("samples", []))}


def transfer_pure_aging_annotation(
    source_scene_id: str,
    target_scene_ids: Sequence[str],
    rotation_degrees_by_target: dict[str, int] | None = None,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """Relabel rotated pure-aging variants from their own bright aging region."""
    state = load_workflow_state(state_path)
    source_record = state.get("samples", {}).get(source_scene_id)
    if source_record is None:
        raise KeyError(f"Unknown workflow scene: {source_scene_id}")
    if source_record.get("annotation_decision") not in {"saved", "accepted"}:
        raise ValueError("The source aging annotation must be saved before transfer.")
    source_root = SCENES_ROOT / source_scene_id
    source_shape = _load_scene_preview_shape(source_scene_id)
    source_masks = {
        name: np.asarray(Image.open(source_root / "masks" / f"{name}.png").convert("L"), dtype=np.uint8)
        for name in ("paint", "pollution", "aging")
    }
    if not np.any(source_masks["aging"] > 0):
        raise ValueError("The source scene has no positive aging pixels.")
    if np.any(source_masks["paint"] > 0) or np.any(source_masks["pollution"] > 0):
        raise ValueError("Pure-aging transfer requires paint and pollution to remain empty.")

    changed: list[str] = []
    rotation_degrees_by_target = rotation_degrees_by_target or {}
    for target_scene_id in target_scene_ids:
        target_record = state.get("samples", {}).get(target_scene_id)
        if target_record is None:
            raise KeyError(f"Unknown workflow scene: {target_scene_id}")
        if target_record.get("background_role") != source_record.get("background_role"):
            raise ValueError("Pure-aging light variants must use the same background role.")
        if _load_scene_preview_shape(target_scene_id) != source_shape:
            raise ValueError("Pure-aging light variants must have the same image shape.")
        target_root = SCENES_ROOT / target_scene_id
        preview = np.asarray(Image.open(target_root / "preview.png").convert("RGB"), dtype=np.uint8)
        bright = np.min(preview, axis=2) > 180
        components, component_count = ndimage.label(bright)
        if component_count == 0:
            raise ValueError(f"No aging candidate was found in {target_scene_id}.")
        component_sizes = np.bincount(components.ravel())
        component_sizes[0] = 0
        aging = components == int(np.argmax(component_sizes))
        aging = ndimage.binary_closing(aging, iterations=6)
        aging = ndimage.binary_dilation(aging, iterations=3)
        source_area = int(np.count_nonzero(source_masks["aging"]))
        target_area = int(np.count_nonzero(aging))
        if not 0.6 * source_area <= target_area <= 1.4 * source_area:
            raise ValueError(
                f"Unsafe aging transfer area for {target_scene_id}: {target_area} vs source {source_area}."
            )
        _write_seed_masks(
            target_root,
            {
                "paint": np.zeros(source_shape, dtype=bool),
                "pollution": np.zeros(source_shape, dtype=bool),
                "aging": aging,
            },
        )
        Image.fromarray(np.zeros(source_shape, dtype=np.uint8)).save(target_root / "masks" / "pigment.png")
        rotation_degrees = int(rotation_degrees_by_target.get(target_scene_id, 0))
        target_record.update(
            {
                "status": "pending_review",
                "stage": "transferred_review",
                "display_prediction_mode": "none",
                "annotation_source": "transferred_from_manual",
                "annotation_source_label": f"参考 {source_scene_id} 语义并按本图旋转位置重新提取，需人工复核",
                "annotation_decision": "pending",
                "parent_scene_id": source_scene_id,
                "annotation_transform": {
                    "rotation_degrees": rotation_degrees,
                    "translation_yx": [0, 0],
                    "method": "target_bright_aging_component",
                },
                "updated_at": _now_iso(),
            }
        )
        changed.append(target_scene_id)
    save_workflow_state(state, state_path)
    manifest = refresh_manifest(state)
    return {
        "action": "transfer_pure_aging_annotation",
        "source_scene_id": source_scene_id,
        "target_scene_ids": changed,
        "manifest_sample_count": len(manifest.get("samples", [])),
    }


def get_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = load_workflow_state(state_path)
    return _state_response(state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="analysis workbench automation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("state")
    subparsers.add_parser("import-latest")
    subparsers.add_parser("import-background4-mothers")
    subparsers.add_parser("import-background4-pure-backgrounds")
    validation_import_parser = subparsers.add_parser("import-validation-capture")
    validation_import_parser.add_argument("--image-path", action="append", required=True)
    validation_import_parser.add_argument("--background-role", required=True, choices=BACKGROUND_ROLES)

    seed_parser = subparsers.add_parser("run-review-seed")
    seed_parser.add_argument("--scene-id", required=True)
    seed_parser.add_argument("--reset", action="store_true")
    background4_predict_parser = subparsers.add_parser("run-background4-prediction")
    background4_predict_parser.add_argument("--scene-id", required=True)
    background4_predict_parser.add_argument(
        "--version-id",
        choices=("background4_v2", "background4_v3", "background4_v3_agingfix_v1_best"),
        default="background4_v2",
    )

    rerun_parser = subparsers.add_parser("rerun-review")
    rerun_parser.add_argument("--scene-id", required=True)

    background_parser = subparsers.add_parser("confirm-background")
    background_parser.add_argument("--scene-id", required=True)
    background_parser.add_argument("--background-role", required=True, choices=BACKGROUND_ROLES)
    background_parser.add_argument("--light-level", choices=LIGHT_LEVELS)

    hold_parser = subparsers.add_parser("hold")
    hold_parser.add_argument("--scene-id", required=True)


    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("--scene-id", required=True)

    batch_approve_parser = subparsers.add_parser("batch-approve-reviewed")
    batch_approve_parser.add_argument("--scene-id", action="append", required=True)

    v3_approve_parser = subparsers.add_parser("approve-background4-v3")
    v3_approve_parser.add_argument("--scene-id", action="append", required=True)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--scene-id", required=True)

    renumber_parser = subparsers.add_parser("renumber")
    renumber_parser.add_argument("--scene-id", required=True)
    renumber_parser.add_argument("--target-scene-id", required=True)

    clear_parser = subparsers.add_parser("clear-annotation")
    clear_parser.add_argument("--scene-id", required=True)

    propagate_parser = subparsers.add_parser("propagate-annotation")
    propagate_parser.add_argument("--scene-id", required=True)

    shift_parser = subparsers.add_parser("shift-annotations")
    shift_parser.add_argument("--scene-id", action="append", required=True)
    shift_parser.add_argument("--shift-y", type=int, required=True)
    shift_parser.add_argument("--shift-x", type=int, default=0)

    aging_transfer_parser = subparsers.add_parser("transfer-pure-aging")
    aging_transfer_parser.add_argument("--source-scene-id", required=True)
    aging_transfer_parser.add_argument("--target-scene-id", action="append", required=True)

    role_parser = subparsers.add_parser("set-background-role")
    role_parser.add_argument("--scene-id", required=True)
    role_parser.add_argument("--background-role", required=True, choices=BACKGROUND_ROLES)


    save_annotation_parser = subparsers.add_parser("save-annotation")
    save_annotation_parser.add_argument("--scene-id", required=True)
    save_annotation_parser.add_argument("--masks-json-path", required=True)


    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "state":
        payload = get_state()
    elif args.command == "import-latest":
        payload = import_latest_capture()
    elif args.command == "import-background4-mothers":
        payload = import_background4_mothers()
    elif args.command == "import-background4-pure-backgrounds":
        payload = import_background4_pure_backgrounds()
    elif args.command == "import-validation-capture":
        payload = import_validation_capture_group([Path(path) for path in args.image_path], args.background_role)
    elif args.command == "run-review-seed":
        payload = run_review_seed(args.scene_id, args.reset)
    elif args.command == "run-background4-prediction":
        payload = run_background4_prediction(args.scene_id, args.version_id)
    elif args.command == "rerun-review":
        payload = rerun_review(args.scene_id)
    elif args.command == "hold":
        payload = hold_scene(args.scene_id)
    elif args.command == "approve":
        payload = approve_scene(args.scene_id)
    elif args.command == "batch-approve-reviewed":
        payload = batch_approve_reviewed_scenes(args.scene_id)
    elif args.command == "approve-background4-v3":
        payload = approve_background4_v3_scenes(args.scene_id)
    elif args.command == "delete":
        payload = delete_scene(args.scene_id)
    elif args.command == "renumber":
        payload = renumber_scene(args.scene_id, args.target_scene_id)
    elif args.command == "clear-annotation":
        payload = clear_scene_annotation(args.scene_id)
    elif args.command == "propagate-annotation":
        payload = propagate_annotation_to_dataset(args.scene_id)
    elif args.command == "shift-annotations":
        payload = shift_annotations(args.scene_id, args.shift_y, args.shift_x)
    elif args.command == "transfer-pure-aging":
        payload = transfer_pure_aging_annotation(args.source_scene_id, args.target_scene_id)
    elif args.command == "set-background-role":
        payload = set_background_role(args.scene_id, args.background_role)
    elif args.command == "save-annotation":
        payload = save_annotation(args.scene_id, json.loads(Path(args.masks_json_path).read_text(encoding="utf-8")))
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

