from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.export_dual_pigment_analysis import export_dual_pigment_analysis
from train.mixed_pigment_subregions import analyze_mixed_paint_region

BAND_LABELS = (450, 550, 600, 650, 700)
HEADS = ("paint", "pollution", "aging")
HEAD_LABELS = {
    "paint": "颜料/绘制主体区域",
    "pollution": "污染/粉末附着区域",
    "aging": "老化/褪变区域",
}
HEAD_COLORS = {
    "paint": "#e84b4b",
    "pollution": "#d8b72a",
    "aging": "#2f8bd8",
}
HEAD_ALPHAS = {
    "paint": 0.28,
    "pollution": 0.42,
    "aging": 0.46,
}
GENERATED_OVERLAY_DIRNAME = "generated_overlays"

PIGMENT_CURVE_SHAPE_WEIGHT = 0.5
PIGMENT_PEAK_MATCH_WEIGHT = 0.3
PIGMENT_SLOPE_WEIGHT = 0.2
PIGMENT_PEAK_RELATIVE_TOLERANCE = 0.03
PIGMENT_REVIEW_MARGIN_THRESHOLD = 0.05
PIGMENT_CLOSE_MARGIN_THRESHOLD = 0.12
PIGMENT_MIN_REVIEW_PIXELS = 64


PIGMENT_SPLIT_RE = "+"


def _split_pigment_tokens(labels: Sequence[str | None]) -> set[str]:
    tokens: set[str] = set()
    for label in labels:
        if not label:
            continue
        for item in str(label).split(PIGMENT_SPLIT_RE):
            token = item.strip()
            if token and token != "???":
                tokens.add(token)
    return tokens


def _derive_preferred_pigments(
    *,
    pigment_prediction: dict[str, object] | None,
    pigment_analysis: dict[str, object],
    sample_label: str | None,
) -> set[str]:
    labels: list[str | None] = [sample_label]
    if pigment_prediction:
        labels.append(str(pigment_prediction.get("predicted_label") or "") or None)
        for candidate in pigment_prediction.get("top_candidates", [])[:2]:
            labels.append(str(candidate.get("name") or "") or None)
    top_candidates = pigment_analysis.get("top_candidates", [])
    if top_candidates:
        labels.append(str(top_candidates[0].get("name") or "") or None)
        if len(top_candidates) > 1 and float(top_candidates[1].get("score", 0.0)) >= float(top_candidates[0].get("score", 0.0)) - 0.08:
            labels.append(str(top_candidates[1].get("name") or "") or None)
    return _split_pigment_tokens(labels)


def normalize_curve(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_value = max(float(value) for value in values)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [float(value) / max_value for value in values]


def compute_head_curve(five_band: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    positive = mask > 0
    positive_pixels = int(positive.sum())
    total_pixels = int(positive.size)
    area_ratio = float(positive_pixels / total_pixels) if total_pixels else 0.0
    if not np.any(positive):
        values = [0.0 for _ in BAND_LABELS]
    else:
        values = [float(five_band[:, :, index][positive].mean()) for index in range(len(BAND_LABELS))]
    normalized = normalize_curve(values)
    peak_index = int(np.argmax(values)) if values else 0
    return {
        "values": values,
        "normalized": normalized,
        "area_ratio": area_ratio,
        "positive_pixels": positive_pixels,
        "total_pixels": total_pixels,
        "peak_wavelength": int(BAND_LABELS[peak_index]),
        "peak_value": float(values[peak_index]) if values else 0.0,
    }


def _asset_path(path: Path, output_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_path.parent.resolve())).as_posix()


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Unsupported hex color: {color}")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _make_overlay_image(
    preview: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    result = preview.astype(np.float32).copy()
    positive = mask > 0
    if bool(np.any(positive)):
        color_array = np.asarray(color, dtype=np.float32)
        result[positive] = ((1.0 - alpha) * result[positive]) + (alpha * color_array)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _annotation_overlay_cache_root(output_path: Path, scene_id: str) -> Path:
    return output_path.parent / GENERATED_OVERLAY_DIRNAME / scene_id


def _atomic_save_image(image: np.ndarray, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target_path.parent, suffix=target_path.suffix, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        Image.fromarray(image).save(temp_path)
        try:
            os.replace(temp_path, target_path)
        except PermissionError:
            if not target_path.exists():
                raise
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _annotation_overlay_output_paths(output_path: Path, scene_id: str) -> dict[str, Path]:
    cache_root = _annotation_overlay_cache_root(output_path, scene_id)
    return {
        "combined": cache_root / "annotation_overlay.png",
        **{head_name: cache_root / f"annotation_{head_name}_overlay.png" for head_name in HEADS},
    }


def _overlay_sources_mtime(paths: Sequence[Path]) -> float:
    return max(path.stat().st_mtime for path in paths if path.exists())


def _annotation_overlay_cache_is_fresh(output_paths: dict[str, Path], source_paths: Sequence[Path]) -> bool:
    required_paths = [output_paths["combined"], *[output_paths[head_name] for head_name in HEADS]]
    if not all(path.exists() for path in required_paths):
        return False
    source_mtime = _overlay_sources_mtime(source_paths)
    return min(path.stat().st_mtime for path in required_paths) >= source_mtime


def _build_annotation_source_sample(
    *,
    scene_root: Path,
    prediction_scene_root: Path,
    five_band: np.ndarray,
    output_path: Path,
) -> dict[str, object]:
    preview_path = scene_root / "preview.png"
    preview = np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.uint8)
    head_data: dict[str, object] = {}
    masks_root = scene_root / "masks"
    use_scene_masks = all((masks_root / f"{head_name}.png").exists() for head_name in HEADS)

    if not use_scene_masks:
        empty_mask = np.zeros(preview.shape[:2], dtype=np.uint8)
        for head_name in HEADS:
            head_data[head_name] = {
                **compute_head_curve(five_band, empty_mask),
                "mask": None,
                "overlay": None,
            }
        return {
            "available": False,
            "assets": {
                "preview": _asset_path(preview_path, output_path),
                "combined_overlay": None,
            },
            "heads": head_data,
            "paint_mask": empty_mask,
        }

    combined = preview.copy()
    mask_paths = {head_name: (masks_root / f"{head_name}.png") for head_name in HEADS}
    output_paths = _annotation_overlay_output_paths(output_path, scene_root.name)
    source_paths = [preview_path, *mask_paths.values()]
    if not _annotation_overlay_cache_is_fresh(output_paths, source_paths):
        for head_name in HEADS:
            mask = np.asarray(Image.open(mask_paths[head_name]).convert("L"), dtype=np.uint8)
            overlay = _make_overlay_image(
                preview,
                mask,
                _hex_to_rgb(HEAD_COLORS[head_name]),
                alpha=HEAD_ALPHAS[head_name],
            )
            _atomic_save_image(overlay, output_paths[head_name])
            combined = _make_overlay_image(
                combined,
                mask,
                _hex_to_rgb(HEAD_COLORS[head_name]),
                alpha=HEAD_ALPHAS[head_name],
            )
        _atomic_save_image(combined, output_paths["combined"])

    paint_mask = np.zeros(preview.shape[:2], dtype=np.uint8)
    for head_name in HEADS:
        mask_path = mask_paths[head_name]
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if head_name == "paint":
            paint_mask = mask
        head_data[head_name] = {
            **compute_head_curve(five_band, mask),
            "mask": _asset_path(mask_path, output_path),
            "overlay": _asset_path(output_paths[head_name], output_path),
        }
    return {
        "available": True,
        "assets": {
            "preview": _asset_path(preview_path, output_path),
            "combined_overlay": _asset_path(output_paths["combined"], output_path),
        },
        "heads": head_data,
        "paint_mask": paint_mask,
    }


def _discover_scene_ids(scenes_root: Path, prediction_root: Path, scene_ids: Sequence[str] | None) -> list[str]:
    if scene_ids:
        return list(scene_ids)
    ids = []
    for scene_root in sorted(scenes_root.iterdir()):
        if not scene_root.is_dir() or not (scene_root / "five_band.npy").exists():
            continue
        if (prediction_root / scene_root.name / "combined_overlay.png").exists():
            ids.append(scene_root.name)
    return ids

def _parse_scene_prediction_roots(raw_items: Sequence[str] | None) -> dict[str, Path]:
    if not raw_items:
        return {}
    mapping: dict[str, Path] = {}
    for item in raw_items:
        scene_id, separator, root_text = item.partition("=")
        if not separator or not scene_id.strip() or not root_text.strip():
            raise ValueError(f"Invalid scene prediction root mapping: {item!r}. Expected SAMPLE_040=/path/to/predictions")
        mapping[scene_id.strip()] = Path(root_text.strip())
    return mapping


def _load_sample_pigments(sample_record_path: Path | None) -> dict[str, str]:
    if sample_record_path is None or not sample_record_path.exists():
        return {}
    pigments: dict[str, str] = {}
    lines = sample_record_path.read_text(encoding="utf-8").splitlines()
    in_target_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| sample_id ") and "| pigment |" in stripped:
            in_target_table = True
            continue
        if in_target_table and stripped.startswith("## "):
            break
        if not in_target_table or not stripped.startswith("| SAMPLE_"):
            continue
        columns = [part.strip() for part in stripped.strip("|").split("|")]
        if len(columns) < 2:
            continue
        pigments[columns[0]] = columns[1]
    return pigments


def _normalize_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return np.zeros_like(vector)
    return vector / norm


def _compute_curve_shape_score(region_curve: Sequence[float], prototype_curve: Sequence[float]) -> float:
    region = np.asarray(region_curve, dtype=np.float32)
    prototype = np.asarray(prototype_curve, dtype=np.float32)
    if region.size == 0 or prototype.size == 0 or region.shape != prototype.shape:
        return 0.0
    # This is pointwise closeness after normalization, not a higher-order shape descriptor.
    distance = float(np.mean(np.abs(region - prototype)))
    return max(0.0, 1.0 - distance)


def _candidate_peak_indices(curve: np.ndarray, *, tolerance: float = PIGMENT_PEAK_RELATIVE_TOLERANCE) -> list[int]:
    if curve.size == 0:
        return []
    peak_value = float(np.max(curve))
    threshold = peak_value - tolerance
    return [int(index) for index, value in enumerate(curve) if float(value) >= threshold]


def _compute_peak_match_score(region_curve: Sequence[float], prototype_curve: Sequence[float]) -> float:
    region = np.asarray(region_curve, dtype=np.float32)
    prototype = np.asarray(prototype_curve, dtype=np.float32)
    if region.size == 0 or prototype.size == 0 or region.shape != prototype.shape:
        return 0.0
    region_peaks = _candidate_peak_indices(region)
    prototype_peaks = _candidate_peak_indices(prototype)
    if not region_peaks or not prototype_peaks:
        return 0.0
    peak_gap = min(abs(region_index - prototype_index) for region_index in region_peaks for prototype_index in prototype_peaks)
    return {0: 1.0, 1: 0.7, 2: 0.35}.get(peak_gap, 0.0)


def _compute_slope_score(region_curve: Sequence[float], prototype_curve: Sequence[float]) -> float:
    region = np.asarray(region_curve, dtype=np.float32)
    prototype = np.asarray(prototype_curve, dtype=np.float32)
    if region.size < 2 or prototype.size < 2 or region.shape != prototype.shape:
        return 0.0
    region_delta = np.diff(region)
    prototype_delta = np.diff(prototype)
    distance = float(np.mean(np.abs(region_delta - prototype_delta)))
    return max(0.0, 1.0 - distance)


def _compute_composite_pigment_score(region_curve: Sequence[float], prototype_curve: Sequence[float]) -> tuple[float, dict[str, float]]:
    curve_shape_score = _compute_curve_shape_score(region_curve, prototype_curve)
    peak_match_score = _compute_peak_match_score(region_curve, prototype_curve)
    slope_score = _compute_slope_score(region_curve, prototype_curve)
    final_score = (
        (PIGMENT_CURVE_SHAPE_WEIGHT * curve_shape_score)
        + (PIGMENT_PEAK_MATCH_WEIGHT * peak_match_score)
        + (PIGMENT_SLOPE_WEIGHT * slope_score)
    )
    return final_score, {
        'curve_shape_score': curve_shape_score,
        'peak_match_score': peak_match_score,
        'slope_score': slope_score,
    }


def _summarize_candidate_confidence(
    top_candidates: list[dict[str, object]],
    positive_pixels: int,
    total_pixels: int,
) -> dict[str, object]:
    if total_pixels > 1024 and positive_pixels < PIGMENT_MIN_REVIEW_PIXELS:
        return {
            "margin": 0.0,
            "confidence_tier": "review",
            "review_reason": "颜料区域过小，当前只能作为弱提示，建议复核。",
        }
    if not top_candidates:
        return {
            "margin": 0.0,
            "confidence_tier": "review",
            "review_reason": "当前没有可用的颜料候选，建议先补充参考样本。",
        }

    top1 = float(top_candidates[0]["score"])
    top2 = float(top_candidates[1]["score"]) if len(top_candidates) > 1 else 0.0
    margin = top1 - top2
    if margin < PIGMENT_REVIEW_MARGIN_THRESHOLD:
        return {
            "margin": margin,
            "confidence_tier": "review",
            "review_reason": "候选颜料之间过于接近，暂时无法稳定区分，建议复核。",
        }
    if margin < PIGMENT_CLOSE_MARGIN_THRESHOLD:
        return {
            "margin": margin,
            "confidence_tier": "close",
            "review_reason": "当前首选颜料略有领先，但次选仍然接近，建议复核。",
        }
    return {
        "margin": margin,
        "confidence_tier": "clear",
        "review_reason": "当前首选颜料已形成明显领先，可作为较稳定参考。",
    }


def _summarize_pigment_verdict(top_candidates: list[dict[str, object]], positive_pixels: int, total_pixels: int) -> dict[str, object]:
    if total_pixels > 1024 and positive_pixels < PIGMENT_MIN_REVIEW_PIXELS:
        return {
            "verdict": "insufficient",
            "should_review": True,
            "cluster_recommended": False,
            "confidence_gap": None,
            "message": "颜料区域过小，当前候选只能作为弱提示。",
        }
    if not top_candidates:
        return {
            "verdict": "insufficient",
            "should_review": True,
            "cluster_recommended": False,
            "confidence_gap": None,
            "message": "当前没有可用的参考原型，暂时无法进行颜料对比。",
        }

    top1 = top_candidates[0]
    top2 = top_candidates[1] if len(top_candidates) > 1 else None
    gap = float(top1["score"]) - float(top2["score"]) if top2 is not None else None
    if gap is not None and gap < PIGMENT_REVIEW_MARGIN_THRESHOLD:
        return {
            "verdict": "ambiguous",
            "should_review": True,
            "cluster_recommended": True,
            "confidence_gap": gap,
            "message": f"当前区域更接近{top1['name']}，但与{top2['name']}仍然非常接近，建议重点复核。",
        }
    if gap is not None and gap < PIGMENT_CLOSE_MARGIN_THRESHOLD:
        return {
            "verdict": "stable",
            "should_review": True,
            "cluster_recommended": positive_pixels >= 4096,
            "confidence_gap": gap,
            "message": f"当前区域最像{top1['name']}，但与{top2['name']}仍较接近，建议复核。",
        }
    return {
        "verdict": "stable",
        "should_review": False,
        "cluster_recommended": positive_pixels >= 4096,
        "confidence_gap": gap,
        "message": f"当前颜料区域与{top1['name']}最为接近。",
    }


def _build_pigment_prototypes(
    scenes_root: Path,
    pigments: dict[str, str],
    reference_scene_ids: Sequence[str] | None,
) -> dict[str, dict[str, object]]:
    candidate_ids = list(reference_scene_ids) if reference_scene_ids else sorted(pigments)
    values_by_pigment: dict[str, list[list[float]]] = {}
    references_by_pigment: dict[str, list[str]] = {}
    for scene_id in candidate_ids:
        pigment = pigments.get(scene_id)
        if not pigment or pigment == "无颜料" or "+" in pigment:
            continue
        scene_root = scenes_root / scene_id
        if not (scene_root / "five_band.npy").exists():
            continue
        mask_path = scene_root / "masks" / "paint.png"
        if not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if not np.any(mask > 0):
            continue
        five_band = np.load(scene_root / "five_band.npy").astype(np.float32)
        curve = compute_head_curve(five_band, mask)
        values_by_pigment.setdefault(pigment, []).append(curve["values"])
        references_by_pigment.setdefault(pigment, []).append(scene_id)

    prototypes: dict[str, dict[str, object]] = {}
    for pigment, rows in values_by_pigment.items():
        mean_values = np.mean(np.asarray(rows, dtype=np.float32), axis=0)
        mean_values_list = mean_values.tolist()
        prototypes[pigment] = {
            "values": mean_values_list,
            "normalized": normalize_curve(mean_values_list),
            "reference_sample_ids": references_by_pigment[pigment],
            "reference_sample_count": len(references_by_pigment[pigment]),
        }
    return prototypes


def _load_pigment_prediction(sample_prediction_root: Path, output_path: Path) -> dict[str, object] | None:
    summary_path = sample_prediction_root / "pigment_summary.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    pixel_map = sample_prediction_root / "pigment_pred.png"
    if pixel_map.exists():
        payload["pixel_map"] = _asset_path(pixel_map, output_path)
    return payload


def _dual_pigment_outputs_exist(output_root: Path) -> bool:
    return all(
        (output_root / name).exists()
        for name in (
            "dual_pigment_overlay.png",
            "dual_pigment_labels.png",
            "dual_pigment_curves.png",
            "dual_pigment_summary.csv",
            "dual_pigment_summary.json",
        )
    )


def _ensure_dual_pigment_analysis(
    *,
    scene_root: Path,
    pigment_assets_root: Path,
    paint_mask_path: Path,
    sample_record_path: Path | None,
    reference_scenes_root: Path,
) -> None:
    if sample_record_path is None or not sample_record_path.exists() or not paint_mask_path.exists():
        return
    paint_mask = np.asarray(Image.open(paint_mask_path).convert("L"), dtype=np.uint8)
    if int((paint_mask > 0).sum()) < 2:
        return
    output_root = pigment_assets_root / "dual_pigment_analysis"
    if _dual_pigment_outputs_exist(output_root):
        return
    try:
        export_dual_pigment_analysis(
            scene_root=scene_root,
            output_root=output_root,
            sample_record_path=sample_record_path,
            paint_mask_path=paint_mask_path,
            reference_scenes_root=reference_scenes_root,
        )
    except Exception:
        return


def _load_annotation_audit(sample_prediction_root: Path, output_path: Path) -> dict[str, object] | None:
    audit_root = sample_prediction_root / "annotation_audit"
    summary_path = audit_root / "audit_summary.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    heads = payload.get("heads", {})
    for head_name, head_payload in list(heads.items()):
        if not isinstance(head_payload, dict):
            continue
        for candidate_kind in ("missing", "overmark"):
            item = head_payload.get(candidate_kind)
            if not isinstance(item, dict):
                continue
            overlay_name = item.get("overlay")
            mask_name = item.get("mask")
            if overlay_name:
                item["overlay"] = _asset_path(audit_root / str(overlay_name), output_path)
            if mask_name:
                item["mask"] = _asset_path(audit_root / str(mask_name), output_path)
    payload["assets_root"] = str(audit_root)
    return payload


def _build_pigment_analysis(
    *,
    five_band: np.ndarray,
    paint_mask: np.ndarray,
    pigment_assets_root: Path,
    output_path: Path,
    prototypes: dict[str, dict[str, object]],
    scene_root: Path,
    paint_mask_path: Path,
    sample_record_path: Path | None,
    reference_scenes_root: Path,
) -> dict[str, object]:
    region_curve = compute_head_curve(five_band, paint_mask)
    cluster_analysis = {
        "available": False,
        "overlay": None,
        "curve_image": None,
        "summary_csv": None,
    }
    _ensure_dual_pigment_analysis(
        scene_root=scene_root,
        pigment_assets_root=pigment_assets_root,
        paint_mask_path=paint_mask_path,
        sample_record_path=sample_record_path,
        reference_scenes_root=reference_scenes_root,
    )
    cluster_root = pigment_assets_root / "dual_pigment_analysis"
    if not cluster_root.exists():
        cluster_root = pigment_assets_root
    cluster_overlay = cluster_root / "dual_pigment_overlay.png"
    cluster_curve = cluster_root / "dual_pigment_curves.png"
    cluster_summary = cluster_root / "dual_pigment_summary.csv"
    cluster_labels = cluster_root / "dual_pigment_labels.png"
    cluster_summary_json = cluster_root / "dual_pigment_summary.json"
    if cluster_overlay.exists() and cluster_curve.exists() and cluster_summary.exists():
        region_summaries: list[dict[str, object]] = []
        if cluster_summary_json.exists():
            try:
                cluster_payload = json.loads(cluster_summary_json.read_text(encoding="utf-8"))
                region_summaries = list(cluster_payload.get("subregions", []))
            except json.JSONDecodeError:
                region_summaries = []
        cluster_analysis = {
            "available": True,
            "overlay": _asset_path(cluster_overlay, output_path),
            "curve_image": _asset_path(cluster_curve, output_path),
            "summary_csv": _asset_path(cluster_summary, output_path),
            "label_map": _asset_path(cluster_labels, output_path) if cluster_labels.exists() else None,
            "summary_json": _asset_path(cluster_summary_json, output_path) if cluster_summary_json.exists() else None,
            "region_summaries": region_summaries,
        }

    result: dict[str, object] = {
        "enabled": bool(region_curve["positive_pixels"] and prototypes),
        "region_positive_pixels": region_curve["positive_pixels"],
        "region_total_pixels": region_curve["total_pixels"],
        "region_curve": {
            "values": region_curve["values"],
            "normalized": region_curve["normalized"],
            "peak_wavelength": region_curve["peak_wavelength"],
            "peak_value": region_curve["peak_value"],
        },
        "top_candidates": [],
        "verdict": "insufficient",
        "should_review": True,
        "cluster_recommended": False,
        "confidence_gap": None,
        "margin": 0.0,
        "confidence_tier": "review",
        "review_reason": "\u5f53\u524d\u8fd8\u6ca1\u6709\u53ef\u7528\u7684\u989c\u6599\u5019\u9009?",
        "message": "?????????????",
        "cluster_analysis": cluster_analysis,
    }
    if not result["enabled"]:
        return result

    ranked = []
    for pigment_name, prototype in prototypes.items():
        score, score_breakdown = _compute_composite_pigment_score(
            region_curve["normalized"],
            prototype["normalized"],
        )
        ranked.append(
            {
                "name": pigment_name,
                "score": score,
                "score_breakdown": score_breakdown,
                "reference_sample_count": prototype["reference_sample_count"],
                "reference_sample_ids": prototype["reference_sample_ids"],
                "prototype_values": prototype["values"],
                "prototype_normalized": prototype["normalized"],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    result["top_candidates"] = ranked[:3]
    result.update(_summarize_candidate_confidence(result["top_candidates"], int(region_curve["positive_pixels"]), int(region_curve["total_pixels"])))
    result.update(_summarize_pigment_verdict(result["top_candidates"], int(region_curve["positive_pixels"]), int(region_curve["total_pixels"])))
    return result

def _build_samples_for_prediction_root(
    *,
    scenes_root: Path,
    prediction_root: Path,
    output_path: Path,
    sample_pigments: dict[str, str],
    prototypes: dict[str, dict[str, object]],
    sample_record_path: Path | None,
    scene_ids: Sequence[str] | None = None,
    scene_prediction_roots: dict[str, Path] | None = None,
    scene_pigment_roots: dict[str, Path] | None = None,
) -> list[dict[str, object]]:
    scene_prediction_roots = scene_prediction_roots or {}
    scene_pigment_roots = scene_pigment_roots or {}
    samples: list[dict[str, object]] = []
    for scene_id in _discover_scene_ids(scenes_root, prediction_root, scene_ids):
        scene_root = scenes_root / scene_id
        sample_prediction_root = scene_prediction_roots.get(scene_id, prediction_root) / scene_id
        pigment_prediction_root = scene_pigment_roots.get(scene_id, sample_prediction_root.parent) / scene_id
        five_band = np.load(scene_root / "five_band.npy").astype(np.float32)
        if five_band.shape[2] != len(BAND_LABELS):
            raise ValueError(f"{scene_id} expected {len(BAND_LABELS)} bands, got {five_band.shape[2]}.")

        annotation_source = _build_annotation_source_sample(
            scene_root=scene_root,
            prediction_scene_root=sample_prediction_root,
            five_band=five_band,
            output_path=output_path,
        )
        head_data: dict[str, object] = {}
        review_paint_mask = None
        has_prediction = all((sample_prediction_root / f"{head_name}_pred.png").exists() for head_name in HEADS)
        if has_prediction:
            for head_name in HEADS:
                mask_path = sample_prediction_root / f"{head_name}_pred.png"
                mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
                if head_name == "paint":
                    review_paint_mask = mask
                head_data[head_name] = {**compute_head_curve(five_band, mask), "mask": _asset_path(mask_path, output_path), "overlay": _asset_path(sample_prediction_root / f"{head_name}_overlay.png", output_path)}
        else:
            head_data = dict(annotation_source["heads"])
            review_paint_mask = np.asarray(annotation_source["paint_mask"], dtype=np.uint8)
        annotation_paint_mask = np.asarray(annotation_source["paint_mask"], dtype=np.uint8)
        use_annotation_paint_mask = annotation_source.get("available") and bool(np.any(annotation_paint_mask > 0))
        paint_mask = annotation_paint_mask if use_annotation_paint_mask else review_paint_mask
        paint_mask_path = (scene_root / "masks" / "paint.png") if use_annotation_paint_mask else (sample_prediction_root / "paint_pred.png")
        assert paint_mask is not None
        pigment_analysis = _build_pigment_analysis(
            five_band=five_band,
            paint_mask=paint_mask,
            pigment_assets_root=pigment_prediction_root,
            output_path=output_path,
            prototypes=prototypes,
            scene_root=scene_root,
            paint_mask_path=paint_mask_path,
            sample_record_path=sample_record_path,
            reference_scenes_root=scenes_root,
        )
        pigment_analysis["sample_label"] = sample_pigments.get(scene_id)
        pigment_analysis["assets_root"] = str(pigment_prediction_root)
        pigment_prediction = _load_pigment_prediction(pigment_prediction_root, output_path)
        preferred_pigments = _derive_preferred_pigments(
            pigment_prediction=pigment_prediction,
            pigment_analysis=pigment_analysis,
            sample_label=sample_pigments.get(scene_id),
        )
        mixed_pigment_analysis = analyze_mixed_paint_region(
            five_band=five_band,
            paint_mask=paint_mask,
            prototypes=prototypes,
            sample_id=scene_id,
            preferred_pigments=preferred_pigments,
        )

        samples.append(
            {
                "id": scene_id,
                "assets": {
                    "preview": annotation_source["assets"]["preview"],
                    "combined_overlay": (annotation_source["assets"]["combined_overlay"] if annotation_source.get("available") else _asset_path(sample_prediction_root / "combined_overlay.png", output_path)),
                    "review_combined_overlay": (_asset_path(sample_prediction_root / "combined_overlay.png", output_path) if has_prediction else annotation_source["assets"]["combined_overlay"]),
                },
                "heads": (annotation_source["heads"] if annotation_source.get("available") else head_data),
                "review_heads": head_data,
                "annotation_available": bool(annotation_source.get("available")),
                "annotation_assets": annotation_source["assets"],
                "review_assets": {
                    "preview": annotation_source["assets"]["preview"],
                    "combined_overlay": (_asset_path(sample_prediction_root / "combined_overlay.png", output_path) if has_prediction else annotation_source["assets"]["combined_overlay"]),
                },
                "pigment_analysis": pigment_analysis,
                "pigment_prediction": pigment_prediction,
                "mixed_pigment_analysis": mixed_pigment_analysis,
                "annotation_audit": _load_annotation_audit(sample_prediction_root, output_path),
            }
        )
    return samples


def _normalize_version_provenance(prediction_root: Path, version_provenance: dict[str, object] | None) -> dict[str, object]:
    provenance = dict(version_provenance or {})
    provenance.setdefault("prediction_root", str(prediction_root))
    return provenance


def _build_manifest_version(
    *,
    version_id: str,
    version_label: str,
    scenes_root: Path,
    prediction_root: Path,
    output_path: Path,
    sample_pigments: dict[str, str],
    prototypes: dict[str, dict[str, object]],
    sample_record_path: Path | None,
    scene_ids: Sequence[str] | None = None,
    scene_prediction_roots: dict[str, Path] | None = None,
    scene_pigment_roots: dict[str, Path] | None = None,
    version_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    samples = _build_samples_for_prediction_root(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=output_path,
        sample_pigments=sample_pigments,
        prototypes=prototypes,
        sample_record_path=sample_record_path,
        scene_ids=scene_ids,
        scene_prediction_roots=scene_prediction_roots,
        scene_pigment_roots=scene_pigment_roots,
    )
    return {
        "id": version_id,
        "label": version_label,
        "prediction_root": str(prediction_root),
        "provenance": _normalize_version_provenance(prediction_root, version_provenance),
        "sample_count": len(samples),
        "samples": samples,
    }


def build_workbench_manifest(
    scenes_root: Path,
    prediction_root: Path,
    output_path: Path,
    scene_ids: Sequence[str] | None = None,
    sample_record_path: Path | None = None,
    reference_scene_ids: Sequence[str] | None = None,
    scene_prediction_roots: dict[str, Path] | None = None,
    scene_pigment_roots: dict[str, Path] | None = None,
    version_id: str = "default",
    version_label: str = "????",
    version_provenance: dict[str, object] | None = None,
    additional_versions: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    sample_pigments = _load_sample_pigments(sample_record_path)
    prototypes = _build_pigment_prototypes(scenes_root, sample_pigments, reference_scene_ids)

    versions = [
        _build_manifest_version(
            version_id=version_id,
            version_label=version_label,
            scenes_root=scenes_root,
            prediction_root=prediction_root,
            output_path=output_path,
            sample_pigments=sample_pigments,
            prototypes=prototypes,
            sample_record_path=sample_record_path,
            scene_ids=scene_ids,
            scene_prediction_roots=scene_prediction_roots,
            scene_pigment_roots=scene_pigment_roots,
            version_provenance=version_provenance,
        )
    ]

    for item in additional_versions or []:
        extra_prediction_root = Path(str(item["prediction_root"]))
        extra_scene_prediction_roots = _parse_scene_prediction_roots(item.get("scene_prediction_roots")) if item.get("scene_prediction_roots") else {}
        versions.append(
            _build_manifest_version(
                version_id=str(item.get("id") or extra_prediction_root.name),
                version_label=str(item.get("label") or item.get("id") or extra_prediction_root.name),
                scenes_root=scenes_root,
                prediction_root=extra_prediction_root,
                output_path=output_path,
                sample_pigments=sample_pigments,
                prototypes=prototypes,
                sample_record_path=sample_record_path,
                scene_ids=scene_ids,
                scene_prediction_roots=extra_scene_prediction_roots,
                version_provenance=item.get("provenance"),
            )
        )

    active_version = versions[0]
    return {
        "band_labels": list(BAND_LABELS),
        "heads": [
            {"id": head_name, "label": HEAD_LABELS[head_name], "color": HEAD_COLORS[head_name]}
            for head_name in HEADS
        ],
        "current_version_id": active_version["id"],
        "active_version": active_version,
        "versions": versions,
        "samples": active_version["samples"],
    }


def export_workbench_manifest(
    scenes_root: Path,
    prediction_root: Path,
    output_path: Path,
    scene_ids: Sequence[str] | None = None,
    sample_record_path: Path | None = None,
    reference_scene_ids: Sequence[str] | None = None,
    scene_prediction_roots: dict[str, Path] | None = None,
    scene_pigment_roots: dict[str, Path] | None = None,
    version_id: str = "default",
    version_label: str = "????",
    version_provenance: dict[str, object] | None = None,
    additional_versions: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    manifest = build_workbench_manifest(
        scenes_root=scenes_root,
        prediction_root=prediction_root,
        output_path=output_path,
        scene_ids=scene_ids,
        sample_record_path=sample_record_path,
        reference_scene_ids=reference_scene_ids,
        scene_prediction_roots=scene_prediction_roots,
        scene_pigment_roots=scene_pigment_roots,
        version_id=version_id,
        version_label=version_label,
        version_provenance=version_provenance,
        additional_versions=additional_versions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    output_path.write_text(manifest_json + "\n", encoding="utf-8")
    output_path.with_suffix(".js").write_text(
        "window.WORKBENCH_MANIFEST = " + manifest_json + ";\n",
        encoding="utf-8",
    )
    return manifest


def _load_optional_json_file(path_text: str | None) -> object | None:
    if not path_text:
        return None
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出静态 analysis workbench manifest。")
    parser.add_argument("--scenes-root", type=str, required=True)
    parser.add_argument("--prediction-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="ui/analysis_workbench/workbench_manifest.json")
    parser.add_argument("--scene-ids", type=str, nargs="*", default=None)
    parser.add_argument("--sample-record-path", type=str, default=None)
    parser.add_argument("--reference-scene-ids", type=str, nargs="*", default=None)
    parser.add_argument(
        "--scene-prediction-roots",
        type=str,
        nargs="*",
        default=None,
        help="Optional per-scene prediction roots like SAMPLE_040=D:/path/to/predictions",
    )
    parser.add_argument(
        "--scene-pigment-roots",
        type=str,
        nargs="*",
        default=None,
        help="Optional per-scene pigment-analysis roots like SAMPLE_049=D:/path/to/predictions",
    )
    parser.add_argument("--version-id", type=str, default="default")
    parser.add_argument("--version-label", type=str, default="????")
    parser.add_argument(
        "--version-provenance-json",
        type=str,
        default=None,
        help="Optional JSON file with provenance fields such as prediction_root/paint_root/pollution_root/aging_root.",
    )
    parser.add_argument(
        "--additional-versions-json",
        type=str,
        default=None,
        help="Optional JSON file containing a list of extra versions for the workbench selector.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = export_workbench_manifest(
        scenes_root=Path(args.scenes_root),
        prediction_root=Path(args.prediction_root),
        output_path=Path(args.output_path),
        scene_ids=args.scene_ids,
        sample_record_path=Path(args.sample_record_path) if args.sample_record_path else None,
        reference_scene_ids=args.reference_scene_ids,
        scene_prediction_roots=_parse_scene_prediction_roots(args.scene_prediction_roots),
        scene_pigment_roots=_parse_scene_prediction_roots(args.scene_pigment_roots),
        version_id=args.version_id,
        version_label=args.version_label,
        version_provenance=_load_optional_json_file(args.version_provenance_json),
        additional_versions=_load_optional_json_file(args.additional_versions_json),
    )
    print(f"samples={len(manifest['samples'])}")
    print(f"output_path={Path(args.output_path)}")


if __name__ == "__main__":
    main()


