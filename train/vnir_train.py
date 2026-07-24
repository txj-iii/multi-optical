from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from train.five_band_features import (
    augment_five_band_cube_with_spectral_features,
    infer_augmented_five_band_channel_count,
)


PIGMENT_CLASS_NAMES = (
    "无颜料",
    "石绿",
    "石青",
    "朱砂",
    "代赭",
    "石青+朱砂",
    "石青+代赭+朱砂",
)
BACKGROUND4_PIGMENT_CLASS_NAMES = ("朱砂", "代赭", "石青", "石绿")
BACKGROUND4_ROLES = ("代赭", "石青", "石绿", "朱砂")


def append_background_condition_maps(image: torch.Tensor, background_role: str) -> torch.Tensor:
    """Append four constant one-hot maps identifying the known board pigment."""
    if background_role != "unknown" and background_role not in BACKGROUND4_ROLES:
        raise ValueError(f"Unsupported background role: {background_role}")
    channels, height, width = image.shape
    del channels
    maps = torch.zeros((len(BACKGROUND4_ROLES), height, width), dtype=image.dtype)
    if background_role in BACKGROUND4_ROLES:
        maps[BACKGROUND4_ROLES.index(background_role)].fill_(1.0)
    return torch.cat([image, maps], dim=0)


@dataclass(frozen=True)
class SixBandPatchSample:
    patch_name: str
    scene_id: str
    image_path: Path
    paint_mask_path: Path
    pollution_mask_path: Path
    aging_mask_path: Path
    pigment_mask_path: Path | None = None


def sample_has_positive_mask(sample: SixBandPatchSample, required_positive_heads: tuple[str, ...]) -> bool:
    if not required_positive_heads:
        return True
    mask_path_by_head = {
        "paint": sample.paint_mask_path,
        "pollution": sample.pollution_mask_path,
        "aging": sample.aging_mask_path,
    }
    for head_name in required_positive_heads:
        mask_array = np.asarray(Image.open(mask_path_by_head[head_name]).convert("L"), dtype=np.uint8)
        if int((mask_array > 0).sum()) > 0:
            return True
    return False


def extract_scene_id_from_patch_name(patch_name: str) -> str:
    parts = patch_name.split("_")
    if len(parts) >= 2 and parts[0] == "SAMPLE":
        return "_".join(parts[:2])
    return patch_name


def build_sample_pigment_map(sample_record_path: Path) -> tuple[dict[str, int], tuple[str, ...]]:
    sample_to_class: dict[str, int] = {}
    reading_table = False
    for raw_line in sample_record_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("| sample_id ") and "| pigment |" in stripped:
            reading_table = True
            continue
        if not reading_table or not stripped.startswith("| SAMPLE_"):
            continue
        columns = [column.strip() for column in stripped.strip("|").split("|")]
        if len(columns) < 2:
            continue
        sample_id, pigment_name = columns[0], columns[1]
        if pigment_name in PIGMENT_CLASS_NAMES:
            sample_to_class[sample_id] = PIGMENT_CLASS_NAMES.index(pigment_name)
    return sample_to_class, PIGMENT_CLASS_NAMES


def collect_six_band_patch_samples(
    split_root: Path,
    required_positive_heads: tuple[str, ...] = (),
    include_empty_scene_ids: tuple[str, ...] = (),
) -> list[SixBandPatchSample]:
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    samples: list[SixBandPatchSample] = []
    include_empty_scene_id_set = set(include_empty_scene_ids)
    for image_path in sorted(images_dir.glob("*.npy")):
        patch_name = image_path.stem
        scene_id = extract_scene_id_from_patch_name(patch_name)
        patch_mask_root = masks_dir / patch_name
        sample = SixBandPatchSample(
            patch_name=patch_name,
            scene_id=scene_id,
            image_path=image_path,
            paint_mask_path=patch_mask_root / "paint.png",
            pollution_mask_path=patch_mask_root / "pollution.png",
            aging_mask_path=patch_mask_root / "aging.png",
            pigment_mask_path=(patch_mask_root / "pigment.png") if (patch_mask_root / "pigment.png").exists() else None,
        )
        if scene_id in include_empty_scene_id_set or sample_has_positive_mask(sample, required_positive_heads):
            samples.append(sample)
    return samples


def infer_patch_channel_count(samples: list[SixBandPatchSample]) -> int:
    if not samples:
        raise ValueError("Cannot infer channel count from an empty patch sample list.")
    image = np.load(samples[0].image_path, mmap_mode="r")
    if image.ndim != 3:
        raise ValueError(f"Expected patch image to have shape HxWxC, got {image.shape}.")
    return infer_augmented_five_band_channel_count(int(image.shape[2]))


def read_patch_positive_heads(sample: SixBandPatchSample) -> tuple[bool, bool, bool]:
    head_paths = (
        sample.paint_mask_path,
        sample.pollution_mask_path,
        sample.aging_mask_path,
    )
    flags: list[bool] = []
    for path in head_paths:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        flags.append(bool((mask > 0).any()))
    return tuple(flags)  # type: ignore[return-value]


def build_balanced_sample_weights(samples: list[SixBandPatchSample]) -> list[float]:
    if not samples:
        return []
    combination_counts: dict[tuple[bool, bool, bool], int] = {}
    combinations: list[tuple[bool, bool, bool]] = []
    for sample in samples:
        combination = read_patch_positive_heads(sample)
        combinations.append(combination)
        combination_counts[combination] = combination_counts.get(combination, 0) + 1
    return [1.0 / float(combination_counts[combination]) for combination in combinations]


def sample_has_positive_aging(sample: SixBandPatchSample) -> bool:
    mask = np.asarray(Image.open(sample.aging_mask_path).convert("L"), dtype=np.uint8)
    return bool((mask > 0).any())


def build_focus_aging_sample_weights(
    samples: list[SixBandPatchSample],
    focus_scene_ids: tuple[str, ...],
    focus_scene_multiplier: float = 4.0,
    aging_positive_multiplier: float = 2.0,
) -> list[float]:
    focus_scene_id_set = set(focus_scene_ids)
    weights: list[float] = []
    for sample in samples:
        weight = 1.0
        if sample_has_positive_aging(sample):
            weight *= aging_positive_multiplier
            if sample.scene_id in focus_scene_id_set:
                weight *= focus_scene_multiplier
        weights.append(weight)
    return weights


def oversample_focus_aging_samples(
    samples: list[SixBandPatchSample],
    focus_scene_ids: tuple[str, ...],
    focus_positive_repeats: int = 3,
) -> list[SixBandPatchSample]:
    if focus_positive_repeats <= 1 or not focus_scene_ids:
        return samples
    focus_scene_id_set = set(focus_scene_ids)
    expanded = list(samples)
    for sample in samples:
        if sample.scene_id in focus_scene_id_set and sample_has_positive_aging(sample):
            expanded.extend([sample] * (focus_positive_repeats - 1))
    return expanded


class SixBandPatchDataset(Dataset):
    def __init__(
        self,
        samples: list[SixBandPatchSample],
        image_size: int,
        sample_pigment_classes: dict[str, int] | None = None,
        pigment_label_mode: str = "legacy",
        rotation_augmentation: bool = False,
        background_roles: dict[str, str] | None = None,
        background_conditioning: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.sample_pigment_classes = sample_pigment_classes or {}
        self.pigment_label_mode = pigment_label_mode
        self.rotation_augmentation = rotation_augmentation
        self.background_roles = background_roles or {}
        self.background_conditioning = background_conditioning

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_image(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[1:]) == (self.image_size, self.image_size):
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _load_mask_tensor(self, mask_path: Path) -> torch.Tensor:
        mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != (self.image_size, self.image_size):
            mask_image = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        # PIL may expose a read-only view.  Training applies rotations to the
        # resulting tensors, so materialize a writable array first.
        mask_array = np.array(mask_image, dtype=np.uint8, copy=True)
        return torch.from_numpy(mask_array).unsqueeze(0).to(torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = augment_five_band_cube_with_spectral_features(np.load(sample.image_path).astype(np.float32))
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        if self.background_conditioning:
            image_tensor = append_background_condition_maps(image_tensor, self.background_roles[sample.scene_id])
        paint_mask = (self._load_mask_tensor(sample.paint_mask_path) > 0).to(torch.float32)
        pigment_target = int(self.sample_pigment_classes.get(sample.scene_id, 0))
        pigment_valid = float(bool((paint_mask > 0.5).any()) and sample.scene_id in self.sample_pigment_classes)
        pigment_mask = paint_mask
        pigment_labels = torch.full((self.image_size, self.image_size), -100, dtype=torch.long)
        if self.pigment_label_mode == "region4":
            pigment_valid = 0.0
            pigment_mask = torch.zeros_like(paint_mask)
            if sample.pigment_mask_path is not None:
                labels = self._load_mask_tensor(sample.pigment_mask_path).to(torch.long)
                labels = labels * (paint_mask > 0.5).to(torch.long)
                valid = labels[0] > 0
                pigment_labels[valid] = labels[0][valid] - 1
                pigment_valid = float(bool(valid.any()))
        result = {
            "patch_name": sample.patch_name,
            "image": self._resize_image(image_tensor),
            "paint_mask": paint_mask,
            "pollution_mask": (self._load_mask_tensor(sample.pollution_mask_path) > 0).to(torch.float32),
            "aging_mask": (self._load_mask_tensor(sample.aging_mask_path) > 0).to(torch.float32),
            "pigment_target": torch.tensor(pigment_target, dtype=torch.long),
            "pigment_valid": torch.tensor(pigment_valid, dtype=torch.float32),
            "pigment_mask": pigment_mask,
            "pigment_labels": pigment_labels,
        }
        if self.rotation_augmentation:
            turns = index % 4
            if turns:
                for key in ("image", "paint_mask", "pollution_mask", "aging_mask", "pigment_mask", "pigment_labels"):
                    result[key] = torch.rot90(result[key], turns, dims=(-2, -1))
        return result


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> torch.Tensor:
    task_losses = compute_task_losses(outputs, batch)
    return aggregate_multitask_loss(task_losses, {name: 1.0 for name in task_losses})


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def compute_binary_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    pos_weight = torch.tensor([positive_weight], dtype=logits.dtype, device=logits.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice


def compute_pigment_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    valid_mask = valid > 0.5
    if not bool(valid_mask.any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits[valid_mask], targets[valid_mask])


def compute_pigment_pixel_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy only where the human supplied a four-class pigment label."""
    if not bool((labels >= 0).any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits, labels, ignore_index=-100)


def compute_task_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
    paint_positive_weight: float = 1.0,
    pollution_positive_weight: float = 1.0,
    aging_positive_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    paint_mask = batch["paint_mask"]
    pollution_mask = batch["pollution_mask"]
    aging_mask = batch["aging_mask"]
    assert isinstance(paint_mask, torch.Tensor)
    assert isinstance(pollution_mask, torch.Tensor)
    assert isinstance(aging_mask, torch.Tensor)
    losses = {
        "paint": compute_binary_segmentation_loss(
            outputs["paint"],
            paint_mask,
            positive_weight=paint_positive_weight,
        ),
        "pollution": compute_binary_segmentation_loss(
            outputs["pollution"],
            pollution_mask,
            positive_weight=pollution_positive_weight,
        ),
        "aging": compute_binary_segmentation_loss(
            outputs["aging"],
            aging_mask,
            positive_weight=aging_positive_weight,
        ),
    }
    if "pigment" in outputs and "pigment_labels" in batch and outputs["pigment"].ndim == 4:
        pigment_labels = batch["pigment_labels"]
        assert isinstance(pigment_labels, torch.Tensor)
        losses["pigment"] = compute_pigment_pixel_loss(outputs["pigment"], pigment_labels)
    elif "pigment" in outputs and "pigment_target" in batch and "pigment_valid" in batch:
        pigment_target = batch["pigment_target"]
        pigment_valid = batch["pigment_valid"]
        assert isinstance(pigment_target, torch.Tensor)
        assert isinstance(pigment_valid, torch.Tensor)
        losses["pigment"] = compute_pigment_classification_loss(
            outputs["pigment"],
            pigment_target,
            pigment_valid,
        )
    return losses


def compute_competitive_paint_pollution_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> torch.Tensor:
    paint_mask = batch["paint_mask"]
    pollution_mask = batch["pollution_mask"]
    assert isinstance(paint_mask, torch.Tensor)
    assert isinstance(pollution_mask, torch.Tensor)

    paint_positive = paint_mask > 0.5
    pollution_positive = pollution_mask > 0.5
    valid = paint_positive ^ pollution_positive
    if not bool(valid.any()):
        device = outputs["paint"].device
        return torch.zeros((), dtype=torch.float32, device=device)

    logits = torch.cat([outputs["paint"], outputs["pollution"]], dim=1)
    targets = torch.full_like(paint_mask[:, 0, :, :], fill_value=-100, dtype=torch.long)
    targets = torch.where(paint_positive[:, 0, :, :], torch.zeros_like(targets), targets)
    targets = torch.where(pollution_positive[:, 0, :, :], torch.ones_like(targets), targets)
    return F.cross_entropy(logits, targets, ignore_index=-100)


def compute_overlap_suppression_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> torch.Tensor:
    paint_mask = batch["paint_mask"]
    pollution_mask = batch["pollution_mask"]
    assert isinstance(paint_mask, torch.Tensor)
    assert isinstance(pollution_mask, torch.Tensor)

    paint_probability = torch.sigmoid(outputs["paint"])
    pollution_probability = torch.sigmoid(outputs["pollution"])
    target_overlap = (paint_mask > 0.5) & (pollution_mask > 0.5)
    penalty_region = ~target_overlap
    if not bool(penalty_region.any()):
        return torch.zeros((), dtype=paint_probability.dtype, device=paint_probability.device)
    overlap_energy = paint_probability * pollution_probability
    return overlap_energy[penalty_region].mean()


def _dilate_binary_mask(mask: torch.Tensor, edge_width: int) -> torch.Tensor:
    if edge_width <= 0:
        return mask > 0.5
    kernel_size = edge_width * 2 + 1
    return F.max_pool2d((mask > 0.5).to(torch.float32), kernel_size=kernel_size, stride=1, padding=edge_width) > 0.5


def _erode_binary_mask(mask: torch.Tensor, edge_width: int) -> torch.Tensor:
    if edge_width <= 0:
        return mask > 0.5
    kernel_size = edge_width * 2 + 1
    binary = (mask > 0.5).to(torch.float32)
    return (-F.max_pool2d(-binary, kernel_size=kernel_size, stride=1, padding=edge_width)) > 0.5


def compute_aging_paint_edge_suppression_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
    edge_width: int = 2,
) -> torch.Tensor:
    paint_mask = batch["paint_mask"]
    aging_mask = batch["aging_mask"]
    assert isinstance(paint_mask, torch.Tensor)
    assert isinstance(aging_mask, torch.Tensor)

    paint_edge_band = _dilate_binary_mask(paint_mask, edge_width) & ~_erode_binary_mask(paint_mask, edge_width)
    penalty_region = paint_edge_band & ~(aging_mask > 0.5)
    if not bool(penalty_region.any()):
        return torch.zeros((), dtype=outputs["aging"].dtype, device=outputs["aging"].device)
    aging_probability = torch.sigmoid(outputs["aging"])
    return aging_probability[penalty_region].mean()


def _normalize_spectral_variation(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] < 15:
        raise ValueError("Spectral guidance loss requires the 15-channel augmented five-band input.")
    diff_variation = image[:, 5:9, :, :].abs().mean(dim=1, keepdim=True)
    nd_variation = image[:, 13:15, :, :].abs().mean(dim=1, keepdim=True)
    variation = 0.5 * diff_variation + 0.5 * nd_variation
    flat = variation.flatten(start_dim=1)
    min_value = flat.min(dim=1).values[:, None, None, None]
    max_value = flat.max(dim=1).values[:, None, None, None]
    return (variation - min_value) / (max_value - min_value + 1e-6)


def compute_spectral_guidance_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> torch.Tensor:
    image = batch["image"]
    paint_mask = batch["paint_mask"]
    pollution_mask = batch["pollution_mask"]
    assert isinstance(image, torch.Tensor)
    assert isinstance(paint_mask, torch.Tensor)
    assert isinstance(pollution_mask, torch.Tensor)

    variation = _normalize_spectral_variation(image)
    paint_weight = variation.detach()
    pollution_weight = (1.0 - variation).detach()

    paint_loss = F.binary_cross_entropy_with_logits(
        outputs["paint"],
        paint_mask,
        weight=paint_weight,
        reduction="mean",
    )
    pollution_loss = F.binary_cross_entropy_with_logits(
        outputs["pollution"],
        pollution_mask,
        weight=pollution_weight,
        reduction="mean",
    )
    return 0.5 * paint_loss + 0.5 * pollution_loss


def compute_disease_spectral_prior_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> torch.Tensor:
    """Emphasize bright/flat aging and dark/flat pollution without hard thresholds."""
    image = batch["image"]
    aging_mask = batch["aging_mask"]
    pollution_mask = batch["pollution_mask"]
    assert isinstance(image, torch.Tensor)
    assert isinstance(aging_mask, torch.Tensor)
    assert isinstance(pollution_mask, torch.Tensor)
    raw_bands = image[:, :5, :, :]
    brightness = (raw_bands.mean(dim=1, keepdim=True) / 0.3).clamp(0.0, 1.0)
    coefficient_of_variation = raw_bands.std(dim=1, keepdim=True) / (
        raw_bands.mean(dim=1, keepdim=True).abs() + 1e-6
    )
    low_variation = torch.exp(-8.0 * coefficient_of_variation).detach()
    aging_weight = (1.0 + 2.0 * brightness.detach() * low_variation).clamp(1.0, 3.0)
    pollution_weight = (1.0 + 2.0 * (1.0 - brightness.detach()) * low_variation).clamp(1.0, 3.0)
    aging_loss = F.binary_cross_entropy_with_logits(
        outputs["aging"],
        aging_mask,
        weight=aging_weight,
    )
    pollution_loss = F.binary_cross_entropy_with_logits(
        outputs["pollution"],
        pollution_mask,
        weight=pollution_weight,
    )
    return 0.5 * aging_loss + 0.5 * pollution_loss


def compute_exclusive_aging_suppression_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> torch.Tensor:
    """Suppress paint/pollution where the manual label says aging only."""
    paint_mask = batch["paint_mask"]
    pollution_mask = batch["pollution_mask"]
    aging_mask = batch["aging_mask"]
    assert isinstance(paint_mask, torch.Tensor)
    assert isinstance(pollution_mask, torch.Tensor)
    assert isinstance(aging_mask, torch.Tensor)
    exclusive_aging = (aging_mask > 0.5) & ~(paint_mask > 0.5) & ~(pollution_mask > 0.5)
    if not bool(exclusive_aging.any()):
        return torch.zeros((), dtype=outputs["aging"].dtype, device=outputs["aging"].device)
    false_head_energy = torch.sigmoid(outputs["paint"]) + torch.sigmoid(outputs["pollution"])
    return false_head_energy[exclusive_aging].mean()


def compute_dwa_weights(
    task_loss_history: dict[str, list[float]],
    temperature: float = 2.0,
) -> dict[str, float]:
    task_names = list(task_loss_history.keys())
    if any(len(history) < 2 for history in task_loss_history.values()):
        return {name: 1.0 for name in task_names}

    ratios = torch.tensor(
        [
            task_loss_history[name][-1] / max(task_loss_history[name][-2], 1e-8)
            for name in task_names
        ],
        dtype=torch.float32,
    )
    normalized = torch.softmax(ratios / temperature, dim=0) * float(len(task_names))
    return {name: float(normalized[index].item()) for index, name in enumerate(task_names)}


def aggregate_multitask_loss(
    task_losses: dict[str, torch.Tensor],
    task_weights: dict[str, float],
) -> torch.Tensor:
    weighted_losses = [task_losses[name] * task_weights[name] for name in task_losses]
    return torch.stack(weighted_losses).sum() / len(weighted_losses)


def build_six_band_dataloader(
    samples: list[SixBandPatchSample],
    image_size: int,
    batch_size: int,
    sampling_strategy: str = "shuffle",
    focus_scene_ids: tuple[str, ...] = (),
    focus_scene_multiplier: float = 4.0,
    aging_positive_multiplier: float = 2.0,
    sample_pigment_classes: dict[str, int] | None = None,
    pigment_label_mode: str = "legacy",
    rotation_augmentation: bool = False,
    background_roles: dict[str, str] | None = None,
    background_conditioning: bool = False,
    pure_background_scene_ids: tuple[str, ...] = (),
    pure_background_multiplier: float = 1.0,
    scene_group_sampling_targets: dict[str, tuple[set[str], float]] | None = None,
) -> DataLoader:
    dataset = SixBandPatchDataset(
        samples=samples,
        image_size=image_size,
        sample_pigment_classes=sample_pigment_classes,
        pigment_label_mode=pigment_label_mode,
        rotation_augmentation=rotation_augmentation,
        background_roles=background_roles,
        background_conditioning=background_conditioning,
    )
    if sampling_strategy == "balanced":
        values = build_balanced_sample_weights(samples)
        pure_ids = set(pure_background_scene_ids)
        if pure_ids and pure_background_multiplier != 1.0:
            values = [weight * (pure_background_multiplier if sample.scene_id in pure_ids else 1.0) for sample, weight in zip(samples, values)]
        if scene_group_sampling_targets:
            claimed = set().union(*(scene_ids for scene_ids, _ in scene_group_sampling_targets.values()))
            sample_scene_ids = {sample.scene_id for sample in samples}
            if not claimed <= sample_scene_ids:
                raise ValueError(f"Sampling groups contain unknown scenes: {sorted(claimed - sample_scene_ids)}")
            for group_name, (scene_ids, target_fraction) in scene_group_sampling_targets.items():
                indices = [index for index, sample in enumerate(samples) if sample.scene_id in scene_ids]
                current_mass = sum(values[index] for index in indices)
                if current_mass <= 0:
                    raise ValueError(f"Sampling group {group_name} has zero sampling mass.")
                scale = target_fraction / current_mass
                for index in indices:
                    values[index] *= scale
        weights = torch.tensor(values, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    if sampling_strategy == "aging_focus":
        weights = torch.tensor(
            build_focus_aging_sample_weights(
                samples,
                focus_scene_ids=focus_scene_ids,
                focus_scene_multiplier=focus_scene_multiplier,
                aging_positive_multiplier=aging_positive_multiplier,
            ),
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_vnir_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_weights: dict[str, float] | None = None,
    competition_weight: float = 0.0,
    paint_positive_weight: float = 1.0,
    pollution_positive_weight: float = 1.0,
    aging_positive_weight: float = 1.0,
    overlap_penalty_weight: float = 0.0,
    spectral_loss_weight: float = 0.0,
    aging_paint_edge_weight: float = 0.0,
    aging_paint_edge_width: int = 2,
    disease_spectral_weight: float = 0.0,
    aging_exclusivity_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    task_totals: dict[str, float] = {
        "paint": 0.0,
        "pollution": 0.0,
        "aging": 0.0,
        "paint_pollution_competition": 0.0,
        "paint_pollution_overlap": 0.0,
        "spectral_guidance": 0.0,
        "aging_paint_edge": 0.0,
        "disease_spectral_prior": 0.0,
        "aging_exclusivity": 0.0,
    }
    for batch in dataloader:
        images = batch["image"].to(device)
        loss_batch = {
            "image": images,
            "paint_mask": batch["paint_mask"].to(device),
            "pollution_mask": batch["pollution_mask"].to(device),
            "aging_mask": batch["aging_mask"].to(device),
        }
        pigment_mask = batch.get("pigment_mask", loss_batch["paint_mask"])
        assert isinstance(pigment_mask, torch.Tensor)
        outputs = model(images, pigment_mask=pigment_mask.to(device))
        if "pigment_target" in batch and "pigment_valid" in batch:
            loss_batch["pigment_target"] = batch["pigment_target"].to(device)
            loss_batch["pigment_valid"] = batch["pigment_valid"].to(device)
        if "pigment_labels" in batch:
            loss_batch["pigment_labels"] = batch["pigment_labels"].to(device)
        task_losses = compute_task_losses(
            outputs,
            loss_batch,
            paint_positive_weight=paint_positive_weight,
            pollution_positive_weight=pollution_positive_weight,
            aging_positive_weight=aging_positive_weight,
        )
        resolved_task_weights = task_weights or {name: 1.0 for name in task_losses}
        competition_loss = compute_competitive_paint_pollution_loss(outputs, loss_batch)
        overlap_penalty = compute_overlap_suppression_loss(outputs, loss_batch)
        spectral_guidance_loss = (
            compute_spectral_guidance_loss(outputs, loss_batch)
            if spectral_loss_weight > 0.0
            else torch.zeros((), dtype=images.dtype, device=images.device)
        )
        aging_paint_edge_loss = (
            compute_aging_paint_edge_suppression_loss(outputs, loss_batch, edge_width=aging_paint_edge_width)
            if aging_paint_edge_weight > 0.0
            else torch.zeros((), dtype=images.dtype, device=images.device)
        )
        disease_spectral_loss = (
            compute_disease_spectral_prior_loss(outputs, loss_batch)
            if disease_spectral_weight > 0.0
            else torch.zeros((), dtype=images.dtype, device=images.device)
        )
        aging_exclusivity_loss = (
            compute_exclusive_aging_suppression_loss(outputs, loss_batch)
            if aging_exclusivity_weight > 0.0
            else torch.zeros((), dtype=images.dtype, device=images.device)
        )
        loss = (
            aggregate_multitask_loss(task_losses, resolved_task_weights)
            + competition_weight * competition_loss
            + overlap_penalty_weight * overlap_penalty
            + spectral_loss_weight * spectral_guidance_loss
            + aging_paint_edge_weight * aging_paint_edge_loss
            + disease_spectral_weight * disease_spectral_loss
            + aging_exclusivity_weight * aging_exclusivity_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
        for name, task_loss in task_losses.items():
            task_totals.setdefault(name, 0.0)
            task_totals[name] += float(task_loss.item())
        task_totals["paint_pollution_competition"] += float(competition_loss.item())
        task_totals["paint_pollution_overlap"] += float(overlap_penalty.item())
        task_totals["spectral_guidance"] += float(spectral_guidance_loss.item())
        task_totals["aging_paint_edge"] += float(aging_paint_edge_loss.item())
        task_totals["disease_spectral_prior"] += float(disease_spectral_loss.item())
        task_totals["aging_exclusivity"] += float(aging_exclusivity_loss.item())
    task_means = {name: total / max(total_batches, 1) for name, total in task_totals.items()}
    return total_loss / max(total_batches, 1), task_means
