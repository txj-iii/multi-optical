import numpy as np

from train.mixed_pigment_subregions import (
    analyze_mixed_paint_region,
    should_split_paint_region,
)


def test_should_split_paint_region_accepts_large_separable_region() -> None:
    five_band = np.zeros((32, 32, 5), dtype=np.float32)
    five_band[:16, :, :] = np.array([0.2, 0.3, 0.8, 0.5, 0.4], dtype=np.float32)
    five_band[16:, :, :] = np.array([0.8, 0.7, 0.3, 0.2, 0.1], dtype=np.float32)
    paint_mask = np.full((32, 32), 255, dtype=np.uint8)

    assert should_split_paint_region(five_band, paint_mask) is True


def test_should_split_paint_region_rejects_small_region() -> None:
    five_band = np.ones((16, 16, 5), dtype=np.float32)
    paint_mask = np.zeros((16, 16), dtype=np.uint8)
    paint_mask[:8, :8] = 255

    assert should_split_paint_region(five_band, paint_mask) is False


def test_analyze_mixed_paint_region_returns_two_named_subregions() -> None:
    five_band = np.zeros((32, 32, 5), dtype=np.float32)
    five_band[:16, :, :] = np.array([0.15, 0.25, 0.75, 0.45, 0.35], dtype=np.float32)
    five_band[16:, :, :] = np.array([0.75, 0.65, 0.25, 0.18, 0.12], dtype=np.float32)
    paint_mask = np.full((32, 32), 255, dtype=np.uint8)
    prototypes = {
        "石青": {
            "normalized": [0.2, 0.3, 1.0, 0.6, 0.4],
            "values": [0.2, 0.3, 1.0, 0.6, 0.4],
        },
        "朱砂": {
            "normalized": [1.0, 0.9, 0.35, 0.25, 0.2],
            "values": [1.0, 0.9, 0.35, 0.25, 0.2],
        },
    }

    result = analyze_mixed_paint_region(
        five_band=five_band,
        paint_mask=paint_mask,
        prototypes=prototypes,
        sample_id="SAMPLE_029",
    )

    assert result["enabled"] is True
    assert result["triggered"] is True
    assert result["reason"] == "split_applied"
    assert len(result["subregions"]) == 2
    assert {item["label"] for item in result["subregions"]} == {"石青", "朱砂"}


def test_analyze_mixed_paint_region_rejects_same_label_split() -> None:
    five_band = np.zeros((32, 32, 5), dtype=np.float32)
    five_band[:16, :, :] = np.array([0.15, 0.25, 0.75, 0.45, 0.35], dtype=np.float32)
    five_band[16:, :, :] = np.array([0.75, 0.65, 0.25, 0.18, 0.12], dtype=np.float32)
    paint_mask = np.full((32, 32), 255, dtype=np.uint8)
    prototypes = {
        "朱砂": {
            "normalized": [1.0, 0.9, 0.35, 0.25, 0.2],
            "values": [1.0, 0.9, 0.35, 0.25, 0.2],
        },
    }

    result = analyze_mixed_paint_region(
        five_band=five_band,
        paint_mask=paint_mask,
        prototypes=prototypes,
        sample_id="SAMPLE_XXX",
    )

    assert result["triggered"] is False
    assert result["reason"] == "whole_region_single_pigment_clear"



def test_analyze_mixed_paint_region_rejects_clear_single_pigment_region() -> None:
    five_band = np.zeros((32, 32, 5), dtype=np.float32)
    five_band[:, :, :] = np.array([0.12, 0.28, 0.92, 0.54, 0.33], dtype=np.float32)
    five_band[16:, :, :] += np.array([0.01, 0.0, -0.02, 0.01, 0.0], dtype=np.float32)
    paint_mask = np.full((32, 32), 255, dtype=np.uint8)
    prototypes = {
        "??": {
            "normalized": [0.13, 0.30, 1.0, 0.58, 0.36],
            "values": [0.13, 0.30, 1.0, 0.58, 0.36],
        },
        "??": {
            "normalized": [1.0, 0.78, 0.46, 0.23, 0.12],
            "values": [1.0, 0.78, 0.46, 0.23, 0.12],
        },
        "??": {
            "normalized": [0.95, 0.84, 0.31, 0.18, 0.08],
            "values": [0.95, 0.84, 0.31, 0.18, 0.08],
        },
    }

    result = analyze_mixed_paint_region(
        five_band=five_band,
        paint_mask=paint_mask,
        prototypes=prototypes,
        sample_id="SAMPLE_040",
    )

    assert result["triggered"] is False
    assert result["reason"] == "whole_region_single_pigment_clear"


def test_analyze_mixed_paint_region_rejects_weak_subregion_evidence() -> None:
    five_band = np.zeros((32, 32, 5), dtype=np.float32)
    five_band[:16, :, :] = np.array([0.22, 0.34, 0.94, 0.56, 0.33], dtype=np.float32)
    five_band[16:, :, :] = np.array([0.88, 0.74, 0.34, 0.22, 0.14], dtype=np.float32)
    paint_mask = np.full((32, 32), 255, dtype=np.uint8)
    prototypes = {
        "??": {
            "normalized": [0.20, 0.32, 1.0, 0.60, 0.36],
            "values": [0.20, 0.32, 1.0, 0.60, 0.36],
        },
        "????": {
            "normalized": [0.24, 0.35, 1.0, 0.55, 0.31],
            "values": [0.24, 0.35, 1.0, 0.55, 0.31],
        },
        "??": {
            "normalized": [1.0, 0.82, 0.40, 0.22, 0.12],
            "values": [1.0, 0.82, 0.40, 0.22, 0.12],
        },
        "????": {
            "normalized": [0.96, 0.78, 0.36, 0.24, 0.15],
            "values": [0.96, 0.78, 0.36, 0.24, 0.15],
        },
    }

    result = analyze_mixed_paint_region(
        five_band=five_band,
        paint_mask=paint_mask,
        prototypes=prototypes,
        sample_id="SAMPLE_WEAK",
    )

    assert result["triggered"] is False
    assert result["reason"] == "subregion_evidence_weak"


def test_analyze_mixed_paint_region_can_prefer_global_pigment_candidates() -> None:
    five_band = np.zeros((32, 32, 5), dtype=np.float32)
    five_band[:16, :, :] = np.array([0.20, 0.32, 0.92, 0.56, 0.34], dtype=np.float32)
    five_band[16:, :, :] = np.array([0.86, 0.72, 0.34, 0.22, 0.14], dtype=np.float32)
    paint_mask = np.full((32, 32), 255, dtype=np.uint8)
    prototypes = {
        "blue": {
            "normalized": [0.18, 0.30, 1.0, 0.60, 0.36],
            "values": [0.18, 0.30, 1.0, 0.60, 0.36],
        },
        "green": {
            "normalized": [0.21, 0.33, 1.0, 0.57, 0.33],
            "values": [0.21, 0.33, 1.0, 0.57, 0.33],
        },
        "red": {
            "normalized": [1.0, 0.84, 0.39, 0.24, 0.14],
            "values": [1.0, 0.84, 0.39, 0.24, 0.14],
        },
    }

    result = analyze_mixed_paint_region(
        five_band=five_band,
        paint_mask=paint_mask,
        prototypes=prototypes,
        sample_id="SAMPLE_GUIDED",
        preferred_pigments={"blue", "red"},
    )

    assert len(result["subregions"]) == 2
    guided = next(item for item in result["subregions"] if item["label"] == "blue")
    assert guided["raw_label"] in {"blue", "green", "red"}
    assert guided["label_source"] == "preferred_candidates"
    assert guided["preferred_top_candidates"][0]["name"] == "blue"
    assert {item["label"] for item in result["subregions"]} == {"blue", "red"}

