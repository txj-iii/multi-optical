from __future__ import annotations

from typing import Sequence

import numpy as np

BAND_LABELS = (450, 550, 600, 650, 700)
MIN_SPLIT_PIXELS = 512
MIN_SUBREGION_PIXELS = 128
MIN_CENTER_DISTANCE = 0.25
MIN_LABEL_MARGIN = 0.05
MIN_STRONG_SUBREGION_MARGIN = 0.08
WHOLE_REGION_CLEAR_MARGIN = 0.10
MIN_SECONDARY_SUBREGION_RATIO = 0.20


def normalize_curve(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_value = max(float(value) for value in values)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [float(value) / max_value for value in values]


def normalize_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return np.zeros_like(vector)
    return vector / norm


def compute_region_curve(five_band: np.ndarray, mask: np.ndarray) -> dict[str, object]:
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


def _initialize_kmeans_centers(samples: np.ndarray, cluster_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first_index = int(rng.integers(0, len(samples)))
    centers = [samples[first_index]]
    while len(centers) < cluster_count:
        distances = np.stack([np.sum((samples - center) ** 2, axis=1) for center in centers], axis=1)
        farthest_index = int(np.argmax(np.min(distances, axis=1)))
        centers.append(samples[farthest_index])
    return np.stack(centers, axis=0).astype(np.float32)


def cluster_paint_region_spectra(
    five_band: np.ndarray,
    paint_mask: np.ndarray,
    *,
    cluster_count: int = 2,
    seed: int = 0,
    max_iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    positive = paint_mask > 0
    if not np.any(positive):
        raise ValueError("paint mask is empty")
    spectra = five_band[positive].astype(np.float32)
    if spectra.shape[0] < cluster_count:
        raise ValueError("not enough paint pixels for clustering")

    centers = _initialize_kmeans_centers(spectra, cluster_count, seed=seed)
    assignments = np.zeros(spectra.shape[0], dtype=np.int32)

    for _ in range(max_iterations):
        distances = np.sum((spectra[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_index in range(cluster_count):
            cluster_pixels = spectra[assignments == cluster_index]
            if cluster_pixels.size > 0:
                centers[cluster_index] = cluster_pixels.mean(axis=0)

    label_map = np.zeros(paint_mask.shape, dtype=np.uint8)
    label_map[positive] = assignments.astype(np.uint8) + 1
    return label_map, centers


def should_split_paint_region(five_band: np.ndarray, paint_mask: np.ndarray) -> bool:
    positive_pixels = int((paint_mask > 0).sum())
    if positive_pixels < MIN_SPLIT_PIXELS:
        return False
    if positive_pixels < MIN_SUBREGION_PIXELS * 2:
        return False
    try:
        _, centers = cluster_paint_region_spectra(five_band, paint_mask, cluster_count=2, seed=0)
    except ValueError:
        return False
    center_distance = float(np.linalg.norm(normalize_vector(centers[0]) - normalize_vector(centers[1])))
    return center_distance >= MIN_CENTER_DISTANCE


def _rank_curve_against_prototypes(
    curve: Sequence[float],
    prototypes: dict[str, dict[str, object]],
    allowed_names: set[str] | None = None,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    normalized_curve = normalize_vector(curve)
    for name, prototype in prototypes.items():
        if allowed_names is not None and name not in allowed_names:
            continue
        prototype_normalized = np.asarray(prototype.get("normalized", []), dtype=np.float32)
        if prototype_normalized.shape != normalized_curve.shape:
            continue
        score = max(0.0, 1.0 - float(np.mean(np.abs(normalized_curve - prototype_normalized))))
        ranked.append((name, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _summarize_ranked_candidates(ranked: list[tuple[str, float]], limit: int = 3) -> list[dict[str, float | str]]:
    return [
        {"name": name, "score": float(score)}
        for name, score in ranked[:limit]
    ]


def _match_center_to_prototype(
    center: np.ndarray,
    prototypes: dict[str, dict[str, object]],
    preferred_pigments: set[str] | None = None,
) -> dict[str, object]:
    ranked = _rank_curve_against_prototypes(center, prototypes)
    if not ranked:
        return {
            "label": "???",
            "raw_label": "???",
            "score": 0.0,
            "margin": 0.0,
            "review_reason": "no_reference_candidates",
            "top_candidates": [],
            "preferred_top_candidates": [],
            "label_source": "none",
        }

    raw_name, raw_score = ranked[0]
    raw_second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    raw_margin = float(raw_score - raw_second_score)
    preferred_ranked: list[tuple[str, float]] = []
    if preferred_pigments:
        preferred_ranked = _rank_curve_against_prototypes(center, prototypes, allowed_names=preferred_pigments)

    if preferred_ranked:
        top_name, top_score = preferred_ranked[0]
        second_score = preferred_ranked[1][1] if len(preferred_ranked) > 1 else 0.0
        margin = float(top_score - second_score)
        label_source = "preferred_candidates"
    else:
        top_name, top_score = raw_name, raw_score
        margin = raw_margin
        label_source = "global_candidates"

    review_reason = None if margin >= MIN_LABEL_MARGIN else "top_candidates_too_close"
    return {
        "label": top_name,
        "raw_label": raw_name,
        "score": float(top_score),
        "margin": margin,
        "review_reason": review_reason,
        "top_candidates": _summarize_ranked_candidates(ranked),
        "preferred_top_candidates": _summarize_ranked_candidates(preferred_ranked),
        "label_source": label_source,
    }


def analyze_mixed_paint_region(
    *,
    five_band: np.ndarray,
    paint_mask: np.ndarray,
    prototypes: dict[str, dict[str, object]],
    sample_id: str,
    preferred_pigments: set[str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "enabled": bool(prototypes),
        "triggered": False,
        "reason": "not_triggered",
        "sample_id": sample_id,
        "subregions": [],
        "cluster_count": 0,
        "center_distance": 0.0,
        "preferred_pigments": sorted(preferred_pigments) if preferred_pigments else [],
    }
    if not result["enabled"]:
        result["reason"] = "no_prototypes"
        return result

    whole_region = compute_region_curve(five_band, paint_mask)
    whole_ranked = _rank_curve_against_prototypes(whole_region["normalized"], prototypes)
    if whole_ranked:
        whole_top_name, whole_top_score = whole_ranked[0]
        whole_second_score = whole_ranked[1][1] if len(whole_ranked) > 1 else 0.0
        whole_margin = float(whole_top_score - whole_second_score)
        result["whole_region_candidate"] = {
            "label": whole_top_name,
            "score": float(whole_top_score),
            "margin": whole_margin,
            "top_candidates": _summarize_ranked_candidates(whole_ranked),
        }
        if whole_margin >= WHOLE_REGION_CLEAR_MARGIN:
            result["reason"] = "whole_region_single_pigment_clear"
            return result

    if not should_split_paint_region(five_band, paint_mask):
        return result

    label_map, centers = cluster_paint_region_spectra(five_band, paint_mask, cluster_count=2, seed=0)
    center_distance = float(np.linalg.norm(normalize_vector(centers[0]) - normalize_vector(centers[1])))
    subregions: list[dict[str, object]] = []
    total_positive_pixels = int((paint_mask > 0).sum())
    for cluster_id, center in enumerate(centers, start=1):
        cluster_mask = np.where(label_map == cluster_id, 255, 0).astype(np.uint8)
        positive_pixels = int((cluster_mask > 0).sum())
        if positive_pixels < MIN_SUBREGION_PIXELS:
            result["reason"] = "subregion_too_small"
            return result
        matched = _match_center_to_prototype(center, prototypes, preferred_pigments=preferred_pigments)
        region_curve = compute_region_curve(five_band, cluster_mask)
        subregions.append(
            {
                "cluster_id": cluster_id,
                "label": matched["label"],
                "raw_label": matched["raw_label"],
                "label_source": matched["label_source"],
                "score": matched["score"],
                "margin": matched["margin"],
                "review_reason": matched["review_reason"],
                "top_candidates": matched["top_candidates"],
                "preferred_top_candidates": matched["preferred_top_candidates"],
                "positive_pixels": positive_pixels,
                "area_ratio_within_paint": float(positive_pixels / total_positive_pixels) if total_positive_pixels else 0.0,
                "curve_values": region_curve["values"],
                "curve_normalized": region_curve["normalized"],
                "peak_wavelength": region_curve["peak_wavelength"],
            }
        )

    smallest_ratio = min(float(item["area_ratio_within_paint"]) for item in subregions)
    weakest_margin = min(float(item["margin"]) for item in subregions)
    if smallest_ratio < MIN_SECONDARY_SUBREGION_RATIO or weakest_margin < MIN_STRONG_SUBREGION_MARGIN:
        result["reason"] = "subregion_evidence_weak"
        result["subregions"] = subregions
        result["cluster_count"] = len(subregions)
        result["center_distance"] = center_distance
        return result

    if len({item["label"] for item in subregions}) < 2:
        result["reason"] = "same_label_after_split"
        result["subregions"] = subregions
        result["cluster_count"] = len(subregions)
        result["center_distance"] = center_distance
        return result

    result["triggered"] = True
    result["reason"] = "split_applied"
    result["cluster_count"] = len(subregions)
    result["center_distance"] = center_distance
    result["subregions"] = subregions
    return result
