from __future__ import annotations

from pathlib import Path
import sys
import argparse
import csv
import json
from typing import Sequence

HEAD_NAMES = ("paint", "pollution", "aging")

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.model import build_multitask_model
from train.five_band_features import augment_five_band_cube_with_spectral_features
from train.vnir_train import BACKGROUND4_ROLES
from train.six_band_dataset import detect_board_bbox_from_preview

PIGMENT_MIN_REVIEW_PAINT_PIXELS = 64
PIGMENT_REVIEW_MARGIN_THRESHOLD = 0.08
PIGMENT_CLOSE_MARGIN_THRESHOLD = 0.16

POLLUTION_MIN_COMPONENT_AREA = 4
POLLUTION_MAX_SMOOTH_AREA_RATIO = 0.06
POLLUTION_SMOOTH_MIN_FILL_RATIO = 0.55
POLLUTION_SMOOTH_MAX_RING_DELTA = 20.0
POLLUTION_CORE_THRESHOLD = 0.78
POLLUTION_CORE_DILATION_ITERATIONS = 6
AGING_EDGE_WIDTH = 2
AGING_EDGE_KEEP_THRESHOLD = 0.8


def resolve_inference_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def threshold_prediction(prediction: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prediction >= threshold).astype(np.uint8) * 255


def resolve_paint_pollution_conflict(
    paint_probability: np.ndarray,
    pollution_probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    return resolve_paint_pollution_conflict_with_thresholds(
        paint_probability,
        pollution_probability,
        paint_threshold=threshold,
        pollution_threshold=threshold,
    )


def resolve_paint_pollution_conflict_with_thresholds(
    paint_probability: np.ndarray,
    pollution_probability: np.ndarray,
    paint_threshold: float,
    pollution_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    paint_active = paint_probability >= paint_threshold
    pollution_active = pollution_probability >= pollution_threshold
    both_active = paint_active & pollution_active

    paint_mask = paint_active.copy()
    pollution_mask = pollution_active.copy()

    paint_wins = both_active & (paint_probability >= pollution_probability)
    pollution_wins = both_active & ~paint_wins

    pollution_mask[paint_wins] = False
    paint_mask[pollution_wins] = False

    return paint_mask.astype(np.uint8) * 255, pollution_mask.astype(np.uint8) * 255


def make_overlay_image(preview: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    overlay = preview.copy().astype(np.float32)
    positive = mask > 0
    overlay[positive] = overlay[positive] * 0.5 + np.array(color, dtype=np.float32) * 0.5
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _compute_pollution_component_metrics(
    component: np.ndarray,
    probability: np.ndarray,
    preview: np.ndarray,
) -> dict[str, float]:
    area = int(component.sum())
    ys, xs = np.where(component)
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    bbox_area = max(bbox_width * bbox_height, 1)
    probability_mean = float(probability[component].mean()) if area else 0.0
    roi_area = max(int(component.shape[0] * component.shape[1]), 1)
    preview_float = preview.astype(np.float32)
    dilated = ndi.binary_dilation(component, structure=np.ones((9, 9), dtype=bool))
    surround_ring = dilated & ~component
    ring_delta = 0.0
    if bool(surround_ring.any()):
        component_mean = preview_float[component].mean(axis=0)
        ring_mean = preview_float[surround_ring].mean(axis=0)
        ring_delta = float(np.abs(component_mean - ring_mean).mean())
    return {
        'area': float(area),
        'area_ratio': float(area / roi_area),
        'fill_ratio': float(area / bbox_area),
        'ring_delta': ring_delta,
        'probability_mean': probability_mean,
    }


def filter_pollution_components(
    roi_mask: np.ndarray,
    roi_probability: np.ndarray,
    roi_preview: np.ndarray,
    min_component_area: int = POLLUTION_MIN_COMPONENT_AREA,
    max_smooth_area_ratio: float = POLLUTION_MAX_SMOOTH_AREA_RATIO,
    core_threshold: float = POLLUTION_CORE_THRESHOLD,
) -> np.ndarray:
    binary_mask = roi_mask > 0
    if not bool(binary_mask.any()):
        return roi_mask
    labels, component_count = ndi.label(binary_mask, structure=np.ones((3, 3), dtype=np.uint8))
    filtered = np.zeros_like(binary_mask, dtype=bool)
    for component_index in range(1, component_count + 1):
        component = labels == component_index
        metrics = _compute_pollution_component_metrics(component, roi_probability, roi_preview)
        area = int(metrics['area'])
        if area < min_component_area:
            continue
        looks_like_large_smooth_background = (
            metrics['area_ratio'] >= max_smooth_area_ratio
            and metrics['fill_ratio'] >= POLLUTION_SMOOTH_MIN_FILL_RATIO
            and metrics['ring_delta'] <= POLLUTION_SMOOTH_MAX_RING_DELTA
        )
        if looks_like_large_smooth_background:
            core_component = component & (roi_probability >= core_threshold)
            if bool(core_component.any()):
                core_labels, core_count = ndi.label(core_component, structure=np.ones((3, 3), dtype=np.uint8))
                recovered = np.zeros_like(component, dtype=bool)
                for core_index in range(1, core_count + 1):
                    core = core_labels == core_index
                    if int(core.sum()) < min_component_area:
                        continue
                    grown = ndi.binary_dilation(
                        core,
                        structure=np.ones((3, 3), dtype=bool),
                        iterations=POLLUTION_CORE_DILATION_ITERATIONS,
                    ) & component
                    grown_metrics = _compute_pollution_component_metrics(grown, roi_probability, roi_preview)
                    if (
                        grown_metrics['fill_ratio'] < POLLUTION_SMOOTH_MIN_FILL_RATIO
                        or grown_metrics['ring_delta'] > POLLUTION_SMOOTH_MAX_RING_DELTA
                    ):
                        recovered |= grown
                filtered |= recovered
            continue
        filtered |= component
    return filtered.astype(np.uint8) * 255


def _build_paint_edge_band(paint_mask: np.ndarray, edge_width: int = AGING_EDGE_WIDTH) -> np.ndarray:
    binary_mask = paint_mask > 0
    if edge_width <= 0 or not bool(binary_mask.any()):
        return np.zeros_like(binary_mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndi.binary_dilation(binary_mask, structure=structure, iterations=edge_width)
    eroded = ndi.binary_erosion(binary_mask, structure=structure, iterations=edge_width, border_value=0)
    return dilated & ~eroded


def suppress_aging_near_paint_edges(
    aging_probability: np.ndarray,
    paint_mask: np.ndarray,
    aging_threshold: float,
    edge_width: int = AGING_EDGE_WIDTH,
    keep_threshold: float = AGING_EDGE_KEEP_THRESHOLD,
) -> np.ndarray:
    aging_mask = threshold_prediction(aging_probability, threshold=aging_threshold)
    edge_band = _build_paint_edge_band(paint_mask, edge_width=edge_width)
    if not bool(edge_band.any()):
        return aging_mask
    keep_mask = aging_probability >= keep_threshold
    suppressed = (aging_mask > 0) & (~edge_band | keep_mask)
    return suppressed.astype(np.uint8) * 255


def build_head_masks(
    probabilities: dict[str, np.ndarray],
    paint_threshold: float,
    pollution_threshold: float,
    aging_threshold: float,
    composition_mode: str = "conflict_resolved",
    aging_edge_suppression: bool = True,
    aging_edge_width: int = AGING_EDGE_WIDTH,
    aging_edge_keep_threshold: float = AGING_EDGE_KEEP_THRESHOLD,
) -> dict[str, np.ndarray]:
    masks = {
        "paint": threshold_prediction(probabilities["paint"], threshold=paint_threshold),
        "pollution": threshold_prediction(probabilities["pollution"], threshold=pollution_threshold),
        "aging": threshold_prediction(probabilities["aging"], threshold=aging_threshold),
    }
    if composition_mode == "independent":
        if aging_edge_suppression:
            masks["aging"] = suppress_aging_near_paint_edges(
                probabilities["aging"],
                masks["paint"],
                aging_threshold=aging_threshold,
                edge_width=aging_edge_width,
                keep_threshold=aging_edge_keep_threshold,
            )
        return masks
    if composition_mode != "conflict_resolved":
        raise ValueError("composition_mode must be independent or conflict_resolved")
    paint_mask, pollution_mask = resolve_paint_pollution_conflict_with_thresholds(
        probabilities["paint"],
        probabilities["pollution"],
        paint_threshold=paint_threshold,
        pollution_threshold=pollution_threshold,
    )
    masks["paint"] = paint_mask
    masks["pollution"] = pollution_mask
    if aging_edge_suppression:
        masks["aging"] = suppress_aging_near_paint_edges(
            probabilities["aging"],
            paint_mask,
            aging_threshold=aging_threshold,
            edge_width=aging_edge_width,
            keep_threshold=aging_edge_keep_threshold,
        )
    return masks


def compute_probability_stats(probabilities: np.ndarray, gt_mask: np.ndarray | None = None) -> dict[str, float | int]:
    flat = probabilities.astype(np.float32).reshape(-1)
    stats: dict[str, float | int] = {
        "mean": float(flat.mean()) if flat.size else 0.0,
        "max": float(flat.max()) if flat.size else 0.0,
        "p95": float(np.percentile(flat, 95)) if flat.size else 0.0,
        "p99": float(np.percentile(flat, 99)) if flat.size else 0.0,
        "gt_positive_pixels": 0,
        "positive_gt_mean": 0.0,
        "positive_gt_p95": 0.0,
    }
    if gt_mask is None:
        return stats
    positive = gt_mask > 0
    stats["gt_positive_pixels"] = int(positive.sum())
    if bool(positive.any()):
        positive_values = probabilities[positive].astype(np.float32)
        stats["positive_gt_mean"] = float(positive_values.mean())
        stats["positive_gt_p95"] = float(np.percentile(positive_values, 95))
    return stats


def _write_probability_diagnostics(rows: list[dict[str, float | int | str]], diagnostic_csv: Path) -> None:
    diagnostic_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "head",
        "mean",
        "max",
        "p95",
        "p99",
        "gt_positive_pixels",
        "positive_gt_mean",
        "positive_gt_p95",
    ]
    with diagnostic_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prediction_output_root(project_root: Path, model_variant: str, epochs: int) -> Path:
    return (
        project_root
        / "train"
        / "experiments"
        / "five_band_predictions"
        / model_variant
        / f"epochs_{epochs}"
    )


def _load_bootstrap_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_variant = checkpoint.get("model_variant", "baseline")
    in_channels = int(checkpoint.get("in_channels", 3))
    pigment_class_names = tuple(checkpoint.get("pigment_class_names", ()))
    model = build_multitask_model(
        variant=model_variant,
        encoder_name="resnet18",
        in_channels=in_channels,
        head_names=("paint", "pollution", "aging"),
        use_spectral_se=bool(checkpoint.get("use_spectral_se", False)),
        pigment_class_count=len(pigment_class_names),
        pigment_masked_pooling=bool(checkpoint.get("pigment_masked_pooling", False)),
        pigment_pixelwise=bool(checkpoint.get("pigment_pixelwise", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, checkpoint


def _resolve_scene_roots(scenes_root: Path, scene_ids: Sequence[str] | None = None) -> list[Path]:
    if scene_ids:
        return [
            scenes_root / scene_id
            for scene_id in scene_ids
            if (scenes_root / scene_id).is_dir() and ((scenes_root / scene_id) / "five_band.npy").exists()
        ]
    return sorted(path for path in scenes_root.iterdir() if path.is_dir() and (path / "five_band.npy").exists())


def _resolve_board_roi(scene_root: Path, preview_array: np.ndarray) -> tuple[int, int, int, int]:
    height, width = preview_array.shape[:2]
    saved_roi_path = scene_root / "board_roi.json"
    if saved_roi_path.exists():
        payload = json.loads(saved_roi_path.read_text(encoding="utf-8"))
        bbox = tuple(int(value) for value in payload["bbox"])
        if len(bbox) != 4:
            raise ValueError(f"Invalid saved board ROI: {saved_roi_path}")
        left, top, right, bottom = bbox
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"Saved board ROI is outside image bounds: {saved_roi_path}")
        return left, top, right, bottom
    # Fail closed: never silently promote the entire camera frame to board ROI.
    return detect_board_bbox_from_preview(preview_array)


def append_background_condition_maps(cube: np.ndarray, background_role: str) -> np.ndarray:
    if background_role not in BACKGROUND4_ROLES:
        raise ValueError(f"background4_v2 needs a valid background_role, got {background_role!r}")
    height, width = cube.shape[:2]
    maps = np.zeros((height, width, len(BACKGROUND4_ROLES)), dtype=cube.dtype)
    maps[:, :, BACKGROUND4_ROLES.index(background_role)] = 1.0
    return np.concatenate([cube, maps], axis=-1)


def _iter_tile_bounds(length: int, tile_size: int, tile_stride: int) -> list[tuple[int, int]]:
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, max(length - tile_size + 1, 1), tile_stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(start + tile_size, length)) for start in starts]


def _predict_roi_probabilities(
    model: torch.nn.Module,
    roi_cube: np.ndarray,
    image_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(roi_cube).permute(2, 0, 1).unsqueeze(0)
    tensor = v2.functional.resize(tensor, [image_size, image_size])
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
    return {
        head_name: torch.sigmoid(outputs[head_name])[0, 0].cpu().numpy()
        for head_name in HEAD_NAMES
    }


def _predict_pigment_logits(
    model: torch.nn.Module,
    roi_cube: np.ndarray,
    image_size: int,
    device: torch.device,
) -> np.ndarray | None:
    tensor = torch.from_numpy(roi_cube).permute(2, 0, 1).unsqueeze(0)
    tensor = v2.functional.resize(tensor, [image_size, image_size])
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
    pigment_logits = outputs.get("pigment")
    if pigment_logits is None:
        return None
    return pigment_logits[0].detach().cpu().numpy()


def _classify_pigment_confidence(
    top_candidates: list[dict[str, object]],
    paint_positive_pixels: int,
    paint_total_pixels: int,
) -> tuple[float, str, str]:
    del paint_total_pixels
    top1 = float(top_candidates[0]["score"]) if top_candidates else 0.0
    top2 = float(top_candidates[1]["score"]) if len(top_candidates) > 1 else 0.0
    margin = top1 - top2
    if paint_positive_pixels < PIGMENT_MIN_REVIEW_PAINT_PIXELS:
        return (
            margin,
            "review",
            f"paint 区域过小 (paint_pixels={paint_positive_pixels}, min_pixels={PIGMENT_MIN_REVIEW_PAINT_PIXELS}, margin={margin:.4f})",
        )
    if margin < PIGMENT_REVIEW_MARGIN_THRESHOLD:
        return margin, "review", f"Top1 与 Top2 过近 (margin={margin:.4f})"
    if margin < PIGMENT_CLOSE_MARGIN_THRESHOLD:
        return margin, "close", f"候选接近，建议复核 (margin={margin:.4f})"
    return margin, "clear", f"Top1 优势明显 (margin={margin:.4f})"

def _build_pigment_summary(
    pigment_logits: np.ndarray,
    pigment_class_names: Sequence[str],
    paint_positive_pixels: int,
    paint_total_pixels: int,
) -> dict[str, object]:
    logits = pigment_logits.astype(np.float32)
    shifted = logits - float(logits.max())
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.maximum(exp_values.sum(), 1e-6)
    ranked_indices = list(np.argsort(probabilities)[::-1])
    top_candidates = [
        {
            "name": pigment_class_names[index],
            "score": float(probabilities[index]),
        }
        for index in ranked_indices[: min(3, len(ranked_indices))]
    ]
    margin, confidence_tier, review_reason = _classify_pigment_confidence(
        top_candidates,
        paint_positive_pixels,
        paint_total_pixels,
    )
    return {
        "enabled": True,
        "predicted_label": pigment_class_names[ranked_indices[0]],
        "predicted_score": float(probabilities[ranked_indices[0]]),
        "top_candidates": top_candidates,
        "margin": margin,
        "confidence_tier": confidence_tier,
        "review_reason": review_reason,
        "paint_positive_pixels": paint_positive_pixels,
        "paint_total_pixels": paint_total_pixels,
        "low_confidence": confidence_tier == "review",
    }


def _predict_tiled_roi_probabilities(
    model: torch.nn.Module,
    five_band_roi: np.ndarray,
    image_size: int,
    device: torch.device,
    tile_size: int,
    tile_stride: int,
    background_role: str | None = None,
    background_conditioning: bool = False,
) -> dict[str, np.ndarray]:
    roi_height, roi_width = five_band_roi.shape[:2]
    probability_sums = {
        head_name: np.zeros((roi_height, roi_width), dtype=np.float32)
        for head_name in ("paint", "pollution", "aging")
    }
    probability_counts = np.zeros((roi_height, roi_width), dtype=np.float32)
    y_bounds = _iter_tile_bounds(roi_height, tile_size=tile_size, tile_stride=tile_stride)
    x_bounds = _iter_tile_bounds(roi_width, tile_size=tile_size, tile_stride=tile_stride)
    for y0, y1 in y_bounds:
        for x0, x1 in x_bounds:
            tile_cube = augment_five_band_cube_with_spectral_features(five_band_roi[y0:y1, x0:x1, :])
            if background_conditioning:
                assert background_role is not None
                tile_cube = append_background_condition_maps(tile_cube, background_role)
            tile_probabilities = _predict_roi_probabilities(
                model,
                tile_cube,
                image_size=image_size,
                device=device,
            )
            tile_width = x1 - x0
            tile_height = y1 - y0
            for head_name, probabilities in tile_probabilities.items():
                probability_image = Image.fromarray(probabilities).resize(
                    (tile_width, tile_height),
                    Image.Resampling.BILINEAR,
                )
                probability_sums[head_name][y0:y1, x0:x1] += np.asarray(probability_image, dtype=np.float32)
            probability_counts[y0:y1, x0:x1] += 1.0
    return {
        head_name: probability_sums[head_name] / np.maximum(probability_counts, 1.0)
        for head_name in HEAD_NAMES
    }


def export_five_band_predictions(
    checkpoint_path: Path,
    scenes_root: Path,
    output_root: Path,
    scene_ids: Sequence[str] | None = None,
    threshold: float = 0.5,
    paint_threshold: float | None = None,
    pollution_threshold: float | None = None,
    aging_threshold: float | None = None,
    diagnostic_csv: Path | None = None,
    save_aging_probability_map: bool = False,
    tile_size: int | None = None,
    tile_stride: int | None = None,
    export_heads: Sequence[str] = HEAD_NAMES,
    composition_mode: str = "conflict_resolved",
    pollution_shape_filter: bool = False,
    pollution_max_smooth_area_ratio: float = POLLUTION_MAX_SMOOTH_AREA_RATIO,
    pollution_core_threshold: float = POLLUTION_CORE_THRESHOLD,
    aging_edge_suppression: bool = True,
    aging_edge_width: int = AGING_EDGE_WIDTH,
    aging_edge_keep_threshold: float = AGING_EDGE_KEEP_THRESHOLD,
    background_role: str | None = None,
) -> list[Path]:
    device = resolve_inference_device()
    print(f"inference_device={device.type}")
    model, checkpoint = _load_bootstrap_model(checkpoint_path, device)
    expected_channels = int(checkpoint.get("in_channels", 3))
    background_conditioning = bool(checkpoint.get("background_conditioning", False))
    if expected_channels != (19 if background_conditioning else 15):
        raise ValueError("This checkpoint does not match the five-band training pipeline.")
    if background_conditioning and background_role not in BACKGROUND4_ROLES:
        raise ValueError("background4_v2 prediction requires background_role from the UI workflow.")

    image_size = int(checkpoint.get("image_size", 128))
    resolved_export_heads = tuple(head_name for head_name in HEAD_NAMES if head_name in set(export_heads))
    if not resolved_export_heads:
        raise ValueError("export_heads must include at least one of paint/pollution/aging")
    resolved_paint_threshold = float(paint_threshold if paint_threshold is not None else threshold)
    resolved_pollution_threshold = float(pollution_threshold if pollution_threshold is not None else threshold)
    resolved_aging_threshold = float(aging_threshold if aging_threshold is not None else threshold)
    output_root.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    for scene_root in _resolve_scene_roots(scenes_root, scene_ids):
        preview_path = scene_root / "preview.png"
        preview_image = Image.open(preview_path).convert("RGB")
        preview_array = np.asarray(preview_image)
        width, height = preview_image.size
        five_band = np.load(scene_root / "five_band.npy").astype(np.float32)
        roi_left, roi_top, roi_right, roi_bottom = _resolve_board_roi(scene_root, preview_array)
        five_band_roi = five_band[roi_top:roi_bottom, roi_left:roi_right, :]
        roi_cube = augment_five_band_cube_with_spectral_features(five_band_roi)
        if background_conditioning:
            roi_cube = append_background_condition_maps(roi_cube, str(background_role))
        roi_height = roi_bottom - roi_top
        roi_width = roi_right - roi_left

        scene_output_dir = output_root / scene_root.name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        color_map = {
            "paint": (255, 0, 0),
            "pollution": (255, 255, 0),
            "aging": (0, 128, 255),
        }
        combined_overlay = preview_array.copy()
        pigment_logits = _predict_pigment_logits(model, roi_cube, image_size=image_size, device=device)
        pigment_class_names = tuple(checkpoint.get("pigment_class_names", ()))
        if tile_size is not None:
            probabilities = _predict_tiled_roi_probabilities(
                model,
                five_band_roi=five_band_roi,
                image_size=image_size,
                device=device,
                tile_size=tile_size,
                tile_stride=tile_stride or tile_size,
                background_role=background_role,
                background_conditioning=background_conditioning,
            )
        else:
            probabilities = _predict_roi_probabilities(
                model,
                roi_cube,
                image_size=image_size,
                device=device,
            )
        head_masks = build_head_masks(
            probabilities,
            paint_threshold=resolved_paint_threshold,
            pollution_threshold=resolved_pollution_threshold,
            aging_threshold=resolved_aging_threshold,
            composition_mode=composition_mode,
            aging_edge_suppression=aging_edge_suppression,
            aging_edge_width=aging_edge_width,
            aging_edge_keep_threshold=aging_edge_keep_threshold,
        )

        for head_name in resolved_export_heads:
            mask = head_masks[head_name]
            probability_image = Image.fromarray(probabilities[head_name]).resize(
                (roi_width, roi_height),
                Image.Resampling.BILINEAR,
            )
            roi_probability_array = np.asarray(probability_image, dtype=np.float32)
            probability_array = np.zeros((height, width), dtype=np.float32)
            probability_array[roi_top:roi_bottom, roi_left:roi_right] = roi_probability_array
            roi_mask_image = Image.fromarray(mask).resize((roi_width, roi_height), Image.Resampling.NEAREST)
            roi_mask_array = np.asarray(roi_mask_image, dtype=np.uint8)
            if head_name == "pollution" and pollution_shape_filter:
                roi_mask_array = filter_pollution_components(
                    roi_mask_array,
                    roi_probability_array,
                    preview_array[roi_top:roi_bottom, roi_left:roi_right, :],
                    max_smooth_area_ratio=pollution_max_smooth_area_ratio,
                    core_threshold=pollution_core_threshold,
                )
            mask_array = np.zeros((height, width), dtype=np.uint8)
            mask_array[roi_top:roi_bottom, roi_left:roi_right] = roi_mask_array
            overlay = make_overlay_image(preview_array, mask_array, color=color_map[head_name])

            mask_path = scene_output_dir / f"{head_name}_pred.png"
            overlay_path = scene_output_dir / f"{head_name}_overlay.png"
            Image.fromarray(mask_array).save(mask_path)
            Image.fromarray(overlay).save(overlay_path)
            exported.extend([mask_path, overlay_path])
            combined_overlay = make_overlay_image(combined_overlay, mask_array, color=color_map[head_name])

            if head_name == "aging" and (diagnostic_csv is not None or save_aging_probability_map):
                gt_path = scene_root / "masks" / "aging.png"
                gt_mask = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8) if gt_path.exists() else None
                stats = compute_probability_stats(probability_array, gt_mask=gt_mask)
                diagnostic_rows.append({"scene": scene_root.name, "head": head_name, **stats})
                if save_aging_probability_map:
                    probability_path = scene_output_dir / "aging_probability.png"
                    Image.fromarray(np.clip(probability_array * 255.0, 0, 255).astype(np.uint8)).save(
                        probability_path
                    )
                    exported.append(probability_path)

        if len(resolved_export_heads) > 1:
            combined_path = scene_output_dir / "combined_overlay.png"
            Image.fromarray(combined_overlay).save(combined_path)
            exported.append(combined_path)

        if pigment_logits is not None and pigment_class_names and "paint" in resolved_export_heads and bool(checkpoint.get("pigment_pixelwise", False)):
            roi_labels = np.argmax(pigment_logits, axis=0).astype(np.uint8) + 1
            roi_labels = np.array(
                Image.fromarray(roi_labels).resize((roi_width, roi_height), Image.Resampling.NEAREST),
                dtype=np.uint8,
                copy=True,
            )
            paint_roi = np.asarray(Image.fromarray(head_masks["paint"]).resize((roi_width, roi_height), Image.Resampling.NEAREST), dtype=np.uint8)
            roi_labels[paint_roi == 0] = 0
            pigment_map = np.zeros((height, width), dtype=np.uint8)
            pigment_map[roi_top:roi_bottom, roi_left:roi_right] = roi_labels
            pigment_path = scene_output_dir / "pigment_pred.png"
            Image.fromarray(pigment_map).save(pigment_path)
            counts = {str(index): int((pigment_map == index).sum()) for index in range(1, len(pigment_class_names) + 1)}
            summary = {"enabled": True, "pixelwise": True, "class_names": list(pigment_class_names), "pixel_counts": counts}
            summary_path = scene_output_dir / "pigment_summary.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            exported.extend([pigment_path, summary_path])
        elif pigment_logits is not None and pigment_class_names and "paint" in resolved_export_heads:
            paint_positive_pixels = int((head_masks["paint"] > 0).sum())
            paint_total_pixels = int(head_masks["paint"].size)
            pigment_summary = _build_pigment_summary(
                pigment_logits,
                pigment_class_names,
                paint_positive_pixels=paint_positive_pixels,
                paint_total_pixels=paint_total_pixels,
            )
            pigment_summary_path = scene_output_dir / "pigment_summary.json"
            pigment_summary_path.write_text(
                json.dumps(pigment_summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            exported.append(pigment_summary_path)

    if diagnostic_csv is not None:
        _write_probability_diagnostics(diagnostic_rows, diagnostic_csv)
        exported.append(diagnostic_csv)

    return exported


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VNIR multitask prediction overlays.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a saved multitask checkpoint.",
    )
    parser.add_argument(
        "--five-band-scenes-root",
        "--scenes-root",
        dest="five_band_scenes_root",
        type=str,
        default=None,
        help="Directory containing scene folders with five_band.npy and preview.png. --scenes-root is an alias.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        default=None,
        help="Optional scene IDs to export, for example SAMPLE_001 SAMPLE_002.",
    )
    parser.add_argument(
        "--background-role",
        choices=BACKGROUND4_ROLES,
        default=None,
        help="Background board role required by background-conditioned checkpoints.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fallback sigmoid threshold used when a per-head threshold is not provided.",
    )
    parser.add_argument(
        "--paint-threshold",
        type=float,
        default=None,
        help="Optional sigmoid threshold override for the paint head.",
    )
    parser.add_argument(
        "--pollution-threshold",
        type=float,
        default=None,
        help="Optional sigmoid threshold override for the pollution head.",
    )
    parser.add_argument(
        "--aging-threshold",
        type=float,
        default=None,
        help="Optional sigmoid threshold override for the aging head.",
    )
    parser.add_argument(
        "--diagnostic-csv",
        type=str,
        default=None,
        help="Optional CSV path for per-scene aging probability diagnostics.",
    )
    parser.add_argument(
        "--save-aging-probability-map",
        action="store_true",
        help="Save an 8-bit aging_probability.png alongside each scene's binary masks.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Optional source-pixel tile size for tiled inference, matching patch training scale.",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=None,
        help="Optional source-pixel tile stride for tiled inference. Defaults to tile size.",
    )
    parser.add_argument(
        "--export-heads",
        nargs="*",
        choices=HEAD_NAMES,
        default=list(HEAD_NAMES),
        help="Optional subset of heads to export. Example: --export-heads aging",
    )
    parser.add_argument(
        "--composition-mode",
        choices=("conflict_resolved", "independent"),
        default="conflict_resolved",
        help="How to combine paint and pollution masks before export. independent keeps all heads separate.",
    )
    parser.add_argument(
        "--pollution-shape-filter",
        action="store_true",
        help="Apply a light connected-component filter that suppresses large smooth pollution regions and keeps irregular granular clusters.",
    )
    parser.add_argument(
        "--pollution-max-smooth-area-ratio",
        type=float,
        default=POLLUTION_MAX_SMOOTH_AREA_RATIO,
        help="Maximum ROI area ratio allowed before a smooth pollution component is treated as likely background.",
    )
    parser.add_argument(
        "--pollution-core-threshold",
        type=float,
        default=POLLUTION_CORE_THRESHOLD,
        help="High-confidence core threshold used to recover irregular pollution inside a large smooth predicted region.",
    )
    parser.add_argument(
        "--disable-aging-edge-suppression",
        action="store_false",
        dest="aging_edge_suppression",
        help="Disable the rule that suppresses weak aging predictions glued to paint edges.",
    )
    parser.add_argument(
        "--aging-edge-width",
        type=int,
        default=AGING_EDGE_WIDTH,
        help="Paint-edge band width used to suppress weak aging glued to paint boundaries.",
    )
    parser.add_argument(
        "--aging-edge-keep-threshold",
        type=float,
        default=AGING_EDGE_KEEP_THRESHOLD,
        help="Keep aging pixels on the paint edge band when their probability reaches this stronger threshold.",
    )
    parser.set_defaults(aging_edge_suppression=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else (
        PROJECT_ROOT
        / "train"
        / "experiments"
        / "five_band_train"
        / "task_specific"
        / "epochs_3"
        / "vnir_multitask_bootstrap_latest.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_variant = checkpoint.get("model_variant", "baseline")
    epochs = int(checkpoint.get("epochs", 1))
    in_channels = int(checkpoint.get("in_channels", 3))
    output_root = (
        Path(args.output_root)
        if args.output_root
        else build_prediction_output_root(PROJECT_ROOT, model_variant, epochs)
    )
    scenes_root = (
        Path(args.five_band_scenes_root)
        if args.five_band_scenes_root
        else (PROJECT_ROOT / "train" / "camera_eval_workspace")
    )
    exported = export_five_band_predictions(
        checkpoint_path,
        scenes_root,
        output_root,
        scene_ids=args.scene_ids,
        threshold=args.threshold,
        paint_threshold=args.paint_threshold,
        pollution_threshold=args.pollution_threshold,
        aging_threshold=args.aging_threshold,
        diagnostic_csv=Path(args.diagnostic_csv) if args.diagnostic_csv else None,
        save_aging_probability_map=args.save_aging_probability_map,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        export_heads=args.export_heads,
        composition_mode=args.composition_mode,
        pollution_shape_filter=args.pollution_shape_filter,
        pollution_max_smooth_area_ratio=args.pollution_max_smooth_area_ratio,
        pollution_core_threshold=args.pollution_core_threshold,
        aging_edge_suppression=args.aging_edge_suppression,
        aging_edge_width=args.aging_edge_width,
        aging_edge_keep_threshold=args.aging_edge_keep_threshold,
        background_role=args.background_role,
    )
    print(f"exported={len(exported)}")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()



