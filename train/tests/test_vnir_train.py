from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train.vnir_train import (
    SixBandPatchDataset,
    aggregate_multitask_loss,
    build_balanced_sample_weights,
    build_focus_aging_sample_weights,
    collect_six_band_patch_samples,
    build_sample_pigment_map,
    compute_competitive_paint_pollution_loss,
    compute_dwa_weights,
    compute_overlap_suppression_loss,
    compute_pigment_classification_loss,
    compute_spectral_guidance_loss,
    compute_multitask_loss,
    compute_task_losses,
    compute_aging_paint_edge_suppression_loss,
    extract_scene_id_from_patch_name,
    infer_patch_channel_count,
    oversample_focus_aging_samples,
    sample_has_positive_mask,
    run_vnir_training_epoch,
)
from train.five_band_features import augment_five_band_cube_with_spectral_features


def test_collect_six_band_patch_samples_reads_npy_and_mask_triplets(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    patch_name = "SCENE_001_x0_y0"
    np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 6), dtype=np.float32))
    patch_mask_root = masks_dir / patch_name
    patch_mask_root.mkdir(parents=True)
    for mask_name in ("paint.png", "pollution.png", "aging.png"):
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / mask_name)

    samples = collect_six_band_patch_samples(split_root)

    assert len(samples) == 1
    assert samples[0].patch_name == patch_name
    assert samples[0].image_path == images_dir / f"{patch_name}.npy"


def test_infer_patch_channel_count_reads_five_band_patch_shape(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    patch_name = "SCENE_001_x0_y0"
    np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 5), dtype=np.float32))
    patch_mask_root = masks_dir / patch_name
    patch_mask_root.mkdir(parents=True)
    for mask_name in ("paint.png", "pollution.png", "aging.png"):
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / mask_name)

    samples = collect_six_band_patch_samples(split_root)

    assert infer_patch_channel_count(samples) == 15


def test_augment_five_band_cube_with_spectral_features_appends_ten_feature_maps() -> None:
    five_band = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 5.0, 5.0, 5.0, 5.0]],
            [[2.0, 4.0, 6.0, 8.0, 10.0], [10.0, 8.0, 6.0, 4.0, 2.0]],
        ],
        dtype=np.float32,
    )

    augmented = augment_five_band_cube_with_spectral_features(five_band)

    assert augmented.shape == (2, 2, 15)
    np.testing.assert_allclose(augmented[:, :, :5], five_band)
    np.testing.assert_allclose(
        augmented[0, 0, 5:],
        np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                2.0 / 1.0,
                3.0 / 2.0,
                4.0 / 3.0,
                5.0 / 4.0,
                (5.0 - 1.0) / (5.0 + 1.0),
                (4.0 - 2.0) / (4.0 + 2.0),
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        augmented[0, 1, 5:],
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )


def test_sample_has_positive_mask_detects_non_empty_requested_heads(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    patch_name = "SCENE_001_x0_y0"
    np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 5), dtype=np.float32))
    patch_mask_root = masks_dir / patch_name
    patch_mask_root.mkdir(parents=True)
    Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(patch_mask_root / "paint.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "pollution.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "aging.png")

    sample = collect_six_band_patch_samples(split_root)[0]

    assert sample_has_positive_mask(sample, ("paint",)) is True
    assert sample_has_positive_mask(sample, ("pollution",)) is False


def test_collect_six_band_patch_samples_can_filter_empty_paint_patches(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    for patch_name, paint_value in (("SCENE_001_x0_y0", 255), ("SCENE_001_x16_y0", 0)):
        np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 5), dtype=np.float32))
        patch_mask_root = masks_dir / patch_name
        patch_mask_root.mkdir(parents=True)
        Image.fromarray(np.full((16, 16), paint_value, dtype=np.uint8)).save(patch_mask_root / "paint.png")
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "pollution.png")
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "aging.png")

    samples = collect_six_band_patch_samples(split_root, required_positive_heads=("paint",))

    assert [sample.patch_name for sample in samples] == ["SCENE_001_x0_y0"]


def test_extract_scene_id_from_patch_name_returns_sample_prefix() -> None:
    assert extract_scene_id_from_patch_name("SAMPLE_015_x421_y31") == "SAMPLE_015"
    assert extract_scene_id_from_patch_name("CAMERA_001_patch") == "CAMERA_001_patch"


def test_collect_six_band_patch_samples_can_keep_empty_negative_control_scene(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    for patch_name in ("SAMPLE_015_x0_y0", "SAMPLE_001_x0_y0"):
        np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 5), dtype=np.float32))
        patch_mask_root = masks_dir / patch_name
        patch_mask_root.mkdir(parents=True)
        for mask_name in ("paint.png", "pollution.png", "aging.png"):
            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / mask_name)

    samples = collect_six_band_patch_samples(
        split_root,
        required_positive_heads=("paint", "pollution", "aging"),
        include_empty_scene_ids=("SAMPLE_015",),
    )

    assert [sample.patch_name for sample in samples] == ["SAMPLE_015_x0_y0"]


def test_build_sample_pigment_map_reads_seven_expected_classes(tmp_path: Path) -> None:
    sample_record = tmp_path / "样本记录规范.md"
    sample_record.write_text(
        "\n".join(
            [
                "| sample_id | pigment | binder | substrate | aging_type | aging_level | pollution_type | pollution_level |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| SAMPLE_001 | 石绿 | x | x | x | x | x | x |",
                "| SAMPLE_004 | 石青 | x | x | x | x | x | x |",
                "| SAMPLE_006 | 朱砂 | x | x | x | x | x | x |",
                "| SAMPLE_009 | 代赭 | x | x | x | x | x | x |",
                "| SAMPLE_015 | 无颜料 | x | x | x | x | x | x |",
                "| SAMPLE_017 | 石青+朱砂 | x | x | x | x | x | x |",
                "| SAMPLE_026 | 石青+代赭+朱砂 | x | x | x | x | x | x |",
            ]
        ),
        encoding="utf-8",
    )

    sample_to_class, class_names = build_sample_pigment_map(sample_record)

    assert class_names == (
        "无颜料",
        "石绿",
        "石青",
        "朱砂",
        "代赭",
        "石青+朱砂",
        "石青+代赭+朱砂",
    )
    assert sample_to_class["SAMPLE_001"] == 1
    assert sample_to_class["SAMPLE_015"] == 0
    assert sample_to_class["SAMPLE_026"] == 6


def test_build_balanced_sample_weights_prioritizes_rare_head_combinations(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    specs = {
        "SCENE_001_x0_y0": {"paint": 255, "pollution": 0, "aging": 0},
        "SCENE_001_x16_y0": {"paint": 255, "pollution": 0, "aging": 0},
        "SCENE_001_x32_y0": {"paint": 0, "pollution": 255, "aging": 0},
    }
    for patch_name, values in specs.items():
        np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 5), dtype=np.float32))
        patch_mask_root = masks_dir / patch_name
        patch_mask_root.mkdir(parents=True)
        for head_name in ("paint", "pollution", "aging"):
            Image.fromarray(np.full((16, 16), values[head_name], dtype=np.uint8)).save(
                patch_mask_root / f"{head_name}.png"
            )

    samples = collect_six_band_patch_samples(split_root)

    weights = build_balanced_sample_weights(samples)

    assert len(weights) == 3
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]


def test_build_focus_aging_sample_weights_prioritizes_focus_scene_aging_patches(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    specs = {
        "SAMPLE_001_x0_y0": {"aging": 255},
        "SAMPLE_024_x0_y0": {"aging": 255},
        "SAMPLE_024_x16_y0": {"aging": 0},
    }
    for patch_name, values in specs.items():
        np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 5), dtype=np.float32))
        patch_mask_root = masks_dir / patch_name
        patch_mask_root.mkdir(parents=True)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "paint.png")
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "pollution.png")
        Image.fromarray(np.full((16, 16), values["aging"], dtype=np.uint8)).save(patch_mask_root / "aging.png")

    samples = collect_six_band_patch_samples(split_root)

    weights = build_focus_aging_sample_weights(
        samples,
        focus_scene_ids=("SAMPLE_024",),
        focus_scene_multiplier=5.0,
        aging_positive_multiplier=2.0,
    )

    assert len(weights) == 3
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]


def test_oversample_focus_aging_samples_repeats_only_focus_positive_patches(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    specs = {
        "SAMPLE_001_x0_y0": {"aging": 255},
        "SAMPLE_024_x0_y0": {"aging": 255},
        "SAMPLE_024_x16_y0": {"aging": 0},
    }
    for patch_name, values in specs.items():
        np.save(images_dir / f"{patch_name}.npy", np.zeros((16, 16, 5), dtype=np.float32))
        patch_mask_root = masks_dir / patch_name
        patch_mask_root.mkdir(parents=True)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "paint.png")
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(patch_mask_root / "pollution.png")
        Image.fromarray(np.full((16, 16), values["aging"], dtype=np.uint8)).save(patch_mask_root / "aging.png")

    samples = collect_six_band_patch_samples(split_root)

    expanded = oversample_focus_aging_samples(
        samples,
        focus_scene_ids=("SAMPLE_024",),
        focus_positive_repeats=3,
    )

    assert [sample.patch_name for sample in expanded].count("SAMPLE_024_x0_y0") == 3
    assert [sample.patch_name for sample in expanded].count("SAMPLE_001_x0_y0") == 1
    assert [sample.patch_name for sample in expanded].count("SAMPLE_024_x16_y0") == 1


def test_competitive_paint_pollution_loss_is_positive_when_heads_disagree_with_labels() -> None:
    outputs = {
        "paint": torch.tensor([[[[2.0, 2.0]]]], dtype=torch.float32),
        "pollution": torch.tensor([[[[2.0, -2.0]]]], dtype=torch.float32),
        "aging": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
    }
    batch = {
        "paint_mask": torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
        "pollution_mask": torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
    }

    loss = compute_competitive_paint_pollution_loss(outputs, batch)

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_overlap_suppression_loss_penalizes_paint_pollution_coactivation() -> None:
    outputs = {
        "paint": torch.tensor([[[[4.0, -4.0]]]], dtype=torch.float32),
        "pollution": torch.tensor([[[[4.0, 4.0]]]], dtype=torch.float32),
        "aging": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
    }
    batch = {
        "paint_mask": torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
        "pollution_mask": torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
    }

    loss = compute_overlap_suppression_loss(outputs, batch)

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0




def test_aging_paint_edge_suppression_loss_penalizes_edge_only_aging_response() -> None:
    outputs = {
        "paint": torch.tensor([[[[8.0, 8.0, -8.0]]]], dtype=torch.float32),
        "pollution": torch.zeros((1, 1, 1, 3), dtype=torch.float32),
        "aging": torch.tensor([[[[2.0, 2.0, -2.0]]]], dtype=torch.float32),
    }
    batch = {
        "paint_mask": torch.tensor([[[[1.0, 1.0, 0.0]]]], dtype=torch.float32),
        "pollution_mask": torch.zeros((1, 1, 1, 3), dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 1, 3), dtype=torch.float32),
    }

    loss = compute_aging_paint_edge_suppression_loss(outputs, batch, edge_width=1)

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_aging_paint_edge_suppression_loss_skips_true_aging_pixels_on_edge() -> None:
    outputs = {
        "paint": torch.zeros((1, 1, 3, 5), dtype=torch.float32),
        "pollution": torch.zeros((1, 1, 3, 5), dtype=torch.float32),
        "aging": torch.full((1, 1, 3, 5), fill_value=-2.0, dtype=torch.float32),
    }
    outputs["paint"][0, 0, :, 1:4] = 8.0
    outputs["aging"][0, 0, :, 1] = 2.0
    negative_batch = {
        "paint_mask": torch.zeros((1, 1, 3, 5), dtype=torch.float32),
        "pollution_mask": torch.zeros((1, 1, 3, 5), dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 3, 5), dtype=torch.float32),
    }
    negative_batch["paint_mask"][0, 0, :, 1:4] = 1.0
    positive_batch = {
        "paint_mask": negative_batch["paint_mask"].clone(),
        "pollution_mask": torch.zeros((1, 1, 3, 5), dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 3, 5), dtype=torch.float32),
    }
    positive_batch["aging_mask"][0, 0, 1, 1] = 1.0

    negative_loss = compute_aging_paint_edge_suppression_loss(outputs, negative_batch, edge_width=1)
    positive_loss = compute_aging_paint_edge_suppression_loss(outputs, positive_batch, edge_width=1)

    assert float(positive_loss.item()) < float(negative_loss.item())
def test_spectral_guidance_loss_prefers_paint_on_high_variation_and_pollution_on_low_variation() -> None:
    image = torch.zeros((1, 15, 1, 2), dtype=torch.float32)
    image[:, 5, :, :] = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    image[:, 6, :, :] = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    image[:, 13, :, :] = torch.tensor([[[0.6, 0.0]]], dtype=torch.float32)
    batch = {
        "image": image,
        "paint_mask": torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
        "pollution_mask": torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
    }
    good_outputs = {
        "paint": torch.tensor([[[[4.0, -4.0]]]], dtype=torch.float32),
        "pollution": torch.tensor([[[[-4.0, 4.0]]]], dtype=torch.float32),
        "aging": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
    }
    bad_outputs = {
        "paint": torch.tensor([[[[-4.0, 4.0]]]], dtype=torch.float32),
        "pollution": torch.tensor([[[[4.0, -4.0]]]], dtype=torch.float32),
        "aging": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
    }

    good_loss = compute_spectral_guidance_loss(good_outputs, batch)
    bad_loss = compute_spectral_guidance_loss(bad_outputs, batch)

    assert torch.isfinite(good_loss)
    assert float(good_loss.item()) < float(bad_loss.item())


def test_six_band_dataset_returns_six_channel_image_and_three_masks(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    patch_name = "SCENE_001_x0_y0"
    image = np.random.default_rng(0).random((24, 24, 5), dtype=np.float32)
    np.save(images_dir / f"{patch_name}.npy", image)
    patch_mask_root = masks_dir / patch_name
    patch_mask_root.mkdir(parents=True)
    for mask_name in ("paint.png", "pollution.png", "aging.png"):
        Image.fromarray(np.zeros((24, 24), dtype=np.uint8)).save(patch_mask_root / mask_name)

    samples = collect_six_band_patch_samples(split_root)
    dataset = SixBandPatchDataset(samples=samples, image_size=32)
    item = dataset[0]

    assert item["image"].shape == (15, 32, 32)
    assert item["paint_mask"].shape == (1, 32, 32)
    assert item["pollution_mask"].shape == (1, 32, 32)
    assert item["aging_mask"].shape == (1, 32, 32)


def test_six_band_dataset_returns_pigment_target_only_for_paint_positive_patch(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    images_dir.mkdir(parents=True)
    patch_names = ("SAMPLE_017_x0_y0", "SAMPLE_015_x0_y0")
    paint_values = (255, 0)
    for patch_name, paint_value in zip(patch_names, paint_values):
        image = np.random.default_rng(0).random((24, 24, 5), dtype=np.float32)
        np.save(images_dir / f"{patch_name}.npy", image)
        patch_mask_root = masks_dir / patch_name
        patch_mask_root.mkdir(parents=True)
        Image.fromarray(np.full((24, 24), paint_value, dtype=np.uint8)).save(patch_mask_root / "paint.png")
        Image.fromarray(np.zeros((24, 24), dtype=np.uint8)).save(patch_mask_root / "pollution.png")
        Image.fromarray(np.zeros((24, 24), dtype=np.uint8)).save(patch_mask_root / "aging.png")

    samples = collect_six_band_patch_samples(split_root)
    dataset = SixBandPatchDataset(
        samples=samples,
        image_size=32,
        sample_pigment_classes={"SAMPLE_017": 5, "SAMPLE_015": 0},
    )

    items = {item["patch_name"]: item for item in (dataset[0], dataset[1])}
    positive_item = items["SAMPLE_017_x0_y0"]
    negative_item = items["SAMPLE_015_x0_y0"]

    assert positive_item["pigment_target"].item() == 5
    assert positive_item["pigment_valid"].item() == 1.0
    assert negative_item["pigment_target"].item() == 0
    assert negative_item["pigment_valid"].item() == 0.0


def test_compute_multitask_loss_returns_scalar() -> None:
    outputs = {
        "paint": torch.zeros((2, 1, 8, 8), dtype=torch.float32),
        "pollution": torch.zeros((2, 1, 8, 8), dtype=torch.float32),
        "aging": torch.zeros((2, 1, 8, 8), dtype=torch.float32),
    }
    batch = {
        "paint_mask": torch.ones((2, 1, 8, 8), dtype=torch.float32),
        "pollution_mask": torch.ones((2, 1, 8, 8), dtype=torch.float32),
        "aging_mask": torch.ones((2, 1, 8, 8), dtype=torch.float32),
    }

    loss = compute_multitask_loss(outputs, batch)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_compute_task_losses_returns_named_scalar_losses() -> None:
    outputs = {
        "paint": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "pollution": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "aging": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
    }
    batch = {
        "paint_mask": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "pollution_mask": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
    }

    losses = compute_task_losses(outputs, batch)

    assert set(losses.keys()) == {"paint", "pollution", "aging"}
    assert all(torch.isfinite(loss) and loss.ndim == 0 for loss in losses.values())


def test_compute_task_losses_can_upweight_pollution_positive_pixels() -> None:
    outputs = {
        "paint": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "pollution": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "aging": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
    }
    batch = {
        "paint_mask": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "pollution_mask": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
        "aging_mask": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
    }

    base_losses = compute_task_losses(outputs, batch)
    weighted_losses = compute_task_losses(outputs, batch, pollution_positive_weight=4.0)

    assert float(weighted_losses["pollution"].item()) > float(base_losses["pollution"].item())


def test_compute_task_losses_can_upweight_aging_positive_pixels() -> None:
    outputs = {
        "paint": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "pollution": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "aging": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
    }
    batch = {
        "paint_mask": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "pollution_mask": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "aging_mask": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
    }

    base_losses = compute_task_losses(outputs, batch)
    weighted_losses = compute_task_losses(outputs, batch, aging_positive_weight=4.0)

    assert float(weighted_losses["aging"].item()) > float(base_losses["aging"].item())


def test_compute_pigment_classification_loss_ignores_invalid_samples() -> None:
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 2], dtype=torch.long)
    valid = torch.tensor([0.0, 1.0], dtype=torch.float32)

    loss = compute_pigment_classification_loss(logits, targets, valid)

    expected = torch.nn.functional.cross_entropy(logits[1:2], targets[1:2])
    assert torch.isfinite(loss)
    assert pytest.approx(float(loss.item()), rel=1e-6) == float(expected.item())


def test_compute_dwa_weights_prioritizes_slower_descending_task() -> None:
    task_loss_history = {
        "paint": [0.8, 0.4],
        "pollution": [0.8, 0.7],
        "aging": [0.8, 0.5],
    }

    weights = compute_dwa_weights(task_loss_history, temperature=2.0)

    assert set(weights.keys()) == {"paint", "pollution", "aging"}
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 3.0
    assert weights["pollution"] > weights["paint"]


def test_aggregate_multitask_loss_returns_scalar() -> None:
    task_losses = {
        "paint": torch.tensor(0.2),
        "pollution": torch.tensor(0.4),
        "aging": torch.tensor(0.6),
    }

    loss = aggregate_multitask_loss(task_losses, {"paint": 1.0, "pollution": 1.0, "aging": 1.0})

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert pytest.approx(float(loss.item()), rel=1e-6) == 0.4


class _TinyThreeHeadDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.full((3, 8, 8), float(index), dtype=torch.float32)
        mask = torch.ones((1, 8, 8), dtype=torch.float32)
        return {
            "image": image,
            "paint_mask": mask,
            "pollution_mask": mask,
            "aging_mask": mask,
        }


class _TinyThreeHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.conv(x)
        return {
            "paint": base,
            "pollution": base,
            "aging": base,
        }


def test_run_vnir_training_epoch_returns_task_history() -> None:
    model = _TinyThreeHeadModel()
    loader = DataLoader(_TinyThreeHeadDataset(), batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    average_loss, task_means = run_vnir_training_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        task_weights={"paint": 1.0, "pollution": 1.0, "aging": 1.0},
    )

    assert isinstance(average_loss, float)
    assert set(task_means.keys()) == {
        "paint",
        "pollution",
        "aging",
        "paint_pollution_competition",
        "paint_pollution_overlap",
        "spectral_guidance",
        "aging_paint_edge",
    }
