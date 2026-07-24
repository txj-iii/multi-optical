from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATCH_ROOT_REL = Path("train") / "five_band_patches" / "train_light_v10_balanced_4849_pollution4447"
WORKFLOW_STATE_PATH = PROJECT_ROOT / "ui" / "analysis_workbench" / "workflow_state.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.model import MODEL_VARIANTS, build_multitask_model
from train.transfer import load_rgb_auxiliary_checkpoint
from train.vnir_train import (
    BACKGROUND4_ROLES,
    BACKGROUND4_PIGMENT_CLASS_NAMES,
    build_six_band_dataloader,
    build_sample_pigment_map,
    collect_six_band_patch_samples,
    compute_dwa_weights,
    infer_patch_channel_count,
    run_vnir_training_epoch,
)


def resolve_training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_patch_root(project_root: Path, patch_root: str | None) -> Path:
    if patch_root:
        return Path(patch_root)
    return project_root / DEFAULT_PATCH_ROOT_REL


def load_training_manifest(
    path: Path,
    patch_roots: Sequence[Path],
    expected_version: str = "background4_v1",
) -> tuple[set[str], str, tuple[str, ...], dict[str, tuple[set[str], float]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version_id") != expected_version:
        raise ValueError(f"Training manifest is not for {expected_version}.")
    manifest_root_values = manifest.get("patch_roots")
    if manifest_root_values is None:
        manifest_root_values = [manifest.get("patch_root", "")]
    manifest_roots = {Path(str(value)).resolve() for value in manifest_root_values}
    requested_roots = {root.resolve() for root in patch_roots}
    if manifest_roots != requested_roots:
        raise ValueError(
            f"Manifest patch roots {sorted(map(str, manifest_roots))} "
            f"do not match {sorted(map(str, requested_roots))}."
        )
    patch_names = {str(record["patch_name"]) for record in manifest.get("records", [])}
    if len(patch_names) != int(manifest.get("patch_count", -1)):
        raise ValueError("Training manifest contains duplicate or incomplete patch names.")
    digest = hashlib.sha256()
    for record in manifest.get("records", []):
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    if digest.hexdigest() != manifest.get("records_sha256"):
        raise ValueError("Training manifest checksum does not match its records.")
    pure_background_ids = tuple(str(value) for value in manifest.get("pure_background_scene_ids", []))
    sampling_groups: dict[str, tuple[set[str], float]] = {}
    claimed_scene_ids: set[str] = set()
    total_fraction = 0.0
    for name, record in manifest.get("sampling_groups", {}).items():
        scene_ids = {str(value) for value in record.get("scene_ids", [])}
        fraction = float(record.get("target_fraction", 0.0))
        if not scene_ids or not 0.0 < fraction < 1.0:
            raise ValueError(f"Invalid sampling group {name}.")
        if claimed_scene_ids & scene_ids:
            raise ValueError("Sampling groups must not overlap.")
        claimed_scene_ids.update(scene_ids)
        total_fraction += fraction
        sampling_groups[str(name)] = (scene_ids, fraction)
    if sampling_groups and abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("Sampling group target fractions must sum to 1.")
    return patch_names, digest.hexdigest(), pure_background_ids, sampling_groups


def load_background_roles(scene_ids: set[str]) -> dict[str, str]:
    state = json.loads(WORKFLOW_STATE_PATH.read_text(encoding="utf-8"))
    records = state.get("samples", {})
    roles = {scene_id: "代赭" for scene_id in scene_ids if "SAMPLE_036" <= scene_id <= "SAMPLE_039"}
    for scene_id in scene_ids:
        role = records.get(scene_id, {}).get("background_role")
        if role in BACKGROUND4_ROLES:
            roles[scene_id] = role
    # Historical 050/053/055 remain in the frozen set, but their original
    # background boards were never recorded.  Zero maps keep them usable
    # without inventing a board identity.
    for scene_id in scene_ids - set(roles):
        roles[scene_id] = "unknown"
    return roles


def build_checkpoint_dir(project_root: Path, model_variant: str, epochs: int) -> Path:
    return (
        project_root
        / "train"
        / "experiments"
        / "five_band_train"
        / model_variant
        / f"epochs_{epochs}"
    )


def resolve_experiment_dir(
    project_root: Path,
    model_variant: str,
    epochs: int,
    output_dir: str | None,
) -> Path:
    if output_dir:
        return Path(output_dir)
    return build_checkpoint_dir(project_root, model_variant, epochs)


def build_task_loss_history(tracked_task_names: Sequence[str]) -> dict[str, list[float]]:
    return {
        **{name: [] for name in tracked_task_names},
        "paint_pollution_competition": [],
        "paint_pollution_overlap": [],
        "spectral_guidance": [],
        "aging_paint_edge": [],
        "disease_spectral_prior": [],
        "aging_exclusivity": [],
    }


def build_checkpoint_metadata(
    *,
    model_variant: str,
    loss_weighting: str,
    dwa_temperature: float,
    task_loss_history: dict[str, list[float]],
    task_weight_history: list[dict[str, float]],
    **extra: object,
) -> dict[str, object]:
    return {
        "model_variant": model_variant,
        "loss_weighting": loss_weighting,
        "dwa_temperature": dwa_temperature,
        "task_loss_history": task_loss_history,
        "task_weight_history": task_weight_history,
        **extra,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VNIR multitask training entry point")
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="baseline",
        help="Choose which multitask model variant to train.",
    )
    parser.add_argument(
        "--aux-checkpoint-path",
        type=str,
        default=None,
        help="Optional auxiliary checkpoint path used for encoder transfer.",
    )
    parser.add_argument(
        "--resume-checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint path used to initialize the full multitask model.",
    )
    parser.add_argument(
        "--patch-root",
        type=str,
        default=None,
        help="Five-band patch split root. Defaults to train/five_band_patches/train_light_v10_balanced_4849_pollution4447.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size.")
    parser.add_argument(
        "--loss-weighting",
        choices=("equal", "dwa"),
        default="equal",
        help="How to combine task losses.",
    )
    parser.add_argument(
        "--dwa-temperature",
        type=float,
        default=2.0,
        help="Softmax temperature used by DWA.",
    )
    parser.add_argument(
        "--require-positive-heads",
        nargs="*",
        choices=("paint", "pollution", "aging"),
        default=["paint"],
        help="Keep only patches that contain positive pixels for at least one of these heads.",
    )
    parser.add_argument(
        "--include-empty-scene-ids",
        nargs="*",
        default=[],
        help="Always keep patches from these scene IDs even when all three masks are empty.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("shuffle", "balanced"),
        default="balanced",
        help="How to sample retained patches during training.",
    )
    parser.add_argument(
        "--competition-weight",
        type=float,
        default=0.5,
        help="Extra weight applied to the paint/pollution competition loss.",
    )
    parser.add_argument(
        "--paint-positive-weight",
        type=float,
        default=1.0,
        help="Extra BCE positive weight applied to paint pixels.",
    )
    parser.add_argument(
        "--pollution-positive-weight",
        type=float,
        default=3.0,
        help="Extra BCE positive weight applied to pollution pixels.",
    )
    parser.add_argument(
        "--aging-positive-weight",
        type=float,
        default=1.0,
        help="Extra BCE positive weight applied to aging pixels.",
    )
    parser.add_argument(
        "--overlap-penalty-weight",
        type=float,
        default=0.25,
        help="Extra weight applied to paint/pollution overlap suppression.",
    )
    parser.add_argument(
        "--spectral-loss-weight",
        type=float,
        default=0.05,
        help="Small auxiliary weight for the spectral guidance loss.",
    )
    parser.add_argument(
        "--disable-spectral-se",
        action="store_false",
        dest="use_spectral_se",
        help="Disable the input-channel Spectral SE attention block.",
    )
    parser.add_argument(
        "--sample-record-path",
        type=str,
        default=None,
        help="Optional sample-record markdown used to derive pigment classes. Defaults to readme/样本记录规范.md.",
    )
    parser.add_argument(
        "--pigment-loss-weight",
        type=float,
        default=0.5,
        help="Relative weight applied to the pigment auxiliary classification loss.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override for checkpoints and logs.",
    )
    parser.add_argument(
        "--additional-patch-root",
        action="append",
        default=[],
        help="Additional patch split root. background4_v3 uses this to retain the frozen v2 base set.",
    )
    parser.add_argument(
        "--training-manifest",
        type=str,
        default=None,
        help="Immutable background4_v1 allow-list generated by prepare_background4_training.py.",
    )
    parser.add_argument("--checkpoint-name", default="vnir_multitask_bootstrap_latest.pt")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an additional epoch checkpoint every N epochs; 0 saves only the final checkpoint.",
    )
    parser.add_argument("--pigment-label-mode", choices=("legacy", "region4"), default="legacy")
    parser.add_argument("--background4", action="store_true", help="Train the isolated four-background/four-pigment version.")
    parser.add_argument("--background4-v2", action="store_true", help="Train background-aware four-background/four-pigment v2.")
    parser.add_argument(
        "--background4-v3",
        action="store_true",
        help="Fine-tune background-aware v3 from the v2 checkpoint with frozen base and reviewed incremental patches.",
    )
    parser.add_argument(
        "--background4-v3-agingfix",
        action="store_true",
        help="Fine-tune v3 with white-aging, dark-pollution, and exclusive-aging supervision.",
    )
    parser.add_argument("--disease-spectral-weight", type=float, default=0.0)
    parser.add_argument("--aging-exclusivity-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.set_defaults(use_spectral_se=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    aux_checkpoint = Path(args.aux_checkpoint_path) if args.aux_checkpoint_path else None
    resume_checkpoint = Path(args.resume_checkpoint_path) if args.resume_checkpoint_path else None
    selected_background4_modes = sum(
        (args.background4, args.background4_v2, args.background4_v3, args.background4_v3_agingfix)
    )
    if selected_background4_modes > 1:
        raise ValueError("Choose only one background4 version flag.")
    is_background4 = bool(selected_background4_modes)
    is_v3_family = args.background4_v3 or args.background4_v3_agingfix
    is_background_conditioned = args.background4_v2 or is_v3_family
    version_id = (
        "background4_v3_agingfix_v1"
        if args.background4_v3_agingfix
        else ("background4_v3" if args.background4_v3 else ("background4_v2" if args.background4_v2 else "background4_v1"))
    )
    if is_background4:
        if args.background4 and resume_checkpoint:
            raise ValueError("background4_v1 must start from scratch; --resume-checkpoint-path is not allowed.")
        if is_v3_family:
            required_checkpoint = (
                PROJECT_ROOT
                / "train"
                / "experiments"
                / "five_band_train"
                / "task_specific"
                / ("background4_v3" if args.background4_v3_agingfix else "background4_v2")
                / ("background4_v3.pt" if args.background4_v3_agingfix else "background4_v2.pt")
            )
            if resume_checkpoint is None:
                resume_checkpoint = required_checkpoint
            if resume_checkpoint.resolve() != required_checkpoint.resolve():
                raise ValueError(f"{version_id} must initialize from {required_checkpoint}.")
            if args.patch_root is None and not args.additional_patch_root:
                args.patch_root = str(
                    PROJECT_ROOT / "train" / "five_band_patches" / "background4_v1" / "train"
                )
                args.additional_patch_root = [
                    str(PROJECT_ROOT / "train" / "five_band_patches" / "background4_v3" / "train")
                ]
            if not args.training_manifest:
                args.training_manifest = str(
                    PROJECT_ROOT
                    / "train"
                    / "five_band_patches"
                    / "background4_v3"
                    / "training_manifest.json"
                )
            # v3 is a controlled fine-tune, not a new hyperparameter search.
            # Freeze every v2 training rule so an accidental CLI override
            # cannot trade one segmentation head for another.
            args.model_variant = "baseline"
            args.loss_weighting = "equal"
            args.dwa_temperature = 2.0
            args.sampling_strategy = "balanced"
            args.competition_weight = 0.5
            args.paint_positive_weight = 1.0
            args.pollution_positive_weight = 3.0
            args.aging_positive_weight = 1.0
            args.overlap_penalty_weight = 0.25
            args.spectral_loss_weight = 0.05
            args.use_spectral_se = True
            args.pigment_loss_weight = 0.5
            if args.background4_v3_agingfix:
                args.disease_spectral_weight = 0.1
                args.aging_exclusivity_weight = 0.5
        args.pigment_label_mode = "region4"
        args.checkpoint_name = f"{version_id}.pt"
        args.output_dir = str(PROJECT_ROOT / "train" / "experiments" / "five_band_train" / "task_specific" / version_id)
        if args.checkpoint_every == 0:
            args.checkpoint_every = 1
    print(f"model_variant={args.model_variant}")
    patch_root = resolve_patch_root(PROJECT_ROOT, args.patch_root)
    patch_roots = [patch_root, *(Path(value) for value in args.additional_patch_root)]
    if len({root.resolve() for root in patch_roots}) != len(patch_roots):
        raise ValueError("Duplicate patch roots are not allowed.")
    if is_background4 and not args.training_manifest:
        raise ValueError("background4 training requires --training-manifest to prevent future validation patches entering training.")
    required_positive_heads = tuple(args.require_positive_heads or [])
    include_empty_scene_ids = tuple(args.include_empty_scene_ids or [])
    allowed_patch_names: set[str] | None = None
    manifest_checksum = None
    sampling_groups: dict[str, tuple[set[str], float]] = {}
    if args.training_manifest:
        manifest_path = Path(args.training_manifest)
        manifest_version = (
            "background4_v1"
            if args.background4_v2
            else ("background4_v3" if is_v3_family else version_id)
        )
        allowed_patch_names, manifest_checksum, pure_background_ids, sampling_groups = load_training_manifest(
            manifest_path,
            patch_roots,
            expected_version=manifest_version,
        )
        include_empty_scene_ids = tuple(dict.fromkeys((*include_empty_scene_ids, *pure_background_ids)))
        # The frozen first-round manifest is already quality-checked and must
        # retain every listed patch, including background-only and small-label
        # context patches.  Do not apply the legacy positive-paint filter.
        required_positive_heads = ()
    samples = []
    for current_patch_root in patch_roots:
        samples.extend(
            collect_six_band_patch_samples(
                current_patch_root,
                required_positive_heads=required_positive_heads,
                include_empty_scene_ids=include_empty_scene_ids,
            )
        )
    patch_names = [sample.patch_name for sample in samples]
    if len(patch_names) != len(set(patch_names)):
        raise ValueError("Patch names must be unique across all patch roots.")
    if allowed_patch_names is not None:
        samples = [sample for sample in samples if sample.patch_name in allowed_patch_names]
        found_patch_names = {sample.patch_name for sample in samples}
        if found_patch_names != allowed_patch_names:
            missing = sorted(allowed_patch_names - found_patch_names)
            raise ValueError(f"Training manifest patches missing after loading: {missing[:5]}")
    sample_record_path = Path(args.sample_record_path) if args.sample_record_path else (PROJECT_ROOT / "readme" / "样本记录规范.md")
    sample_pigment_classes, pigment_class_names = build_sample_pigment_map(sample_record_path)
    if args.pigment_label_mode == "region4":
        sample_pigment_classes = {}
        pigment_class_names = BACKGROUND4_PIGMENT_CLASS_NAMES
    if not samples:
        raise ValueError(
            f"No five-band patches found under {patch_root}. "
            "Run train/six_band_dataset.py export-patches first or pass --patch-root."
        )
    background_roles = load_background_roles({sample.scene_id for sample in samples}) if is_background_conditioned else None
    dataloader = build_six_band_dataloader(
        samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        sampling_strategy=args.sampling_strategy,
        sample_pigment_classes=sample_pigment_classes,
        pigment_label_mode=args.pigment_label_mode,
        rotation_augmentation=is_background4,
        background_roles=background_roles,
        background_conditioning=is_background_conditioned,
        pure_background_scene_ids=pure_background_ids if is_background_conditioned else (),
        pure_background_multiplier=2.5 if is_background_conditioned else 1.0,
        scene_group_sampling_targets=sampling_groups if is_v3_family else None,
    )
    in_channels = infer_patch_channel_count(samples) + (len(BACKGROUND4_ROLES) if is_background_conditioned else 0)
    print("patch_roots=" + ",".join(str(root) for root in patch_roots))
    print(f"patch_count={len(samples)}")
    print(f"in_channels={in_channels}")
    print(f"required_positive_heads={','.join(required_positive_heads) if required_positive_heads else 'none'}")
    print(f"include_empty_scene_ids={','.join(include_empty_scene_ids) if include_empty_scene_ids else 'none'}")
    print(f"sampling_strategy={args.sampling_strategy}")
    print(f"competition_weight={args.competition_weight}")
    print(f"paint_positive_weight={args.paint_positive_weight}")
    print(f"pollution_positive_weight={args.pollution_positive_weight}")
    print(f"aging_positive_weight={args.aging_positive_weight}")
    print(f"overlap_penalty_weight={args.overlap_penalty_weight}")
    print(f"spectral_loss_weight={args.spectral_loss_weight}")
    print(f"disease_spectral_weight={args.disease_spectral_weight}")
    print(f"aging_exclusivity_weight={args.aging_exclusivity_weight}")
    print(f"use_spectral_se={args.use_spectral_se}")
    print(f"pigment_class_count={len(pigment_class_names)}")
    print(f"pigment_loss_weight={args.pigment_loss_weight}")
    resolved_learning_rate = args.learning_rate if args.learning_rate is not None else (
        5e-5 if args.background4_v3_agingfix else (1e-4 if args.background4_v3 else 1e-3)
    )
    print(f"learning_rate={resolved_learning_rate}")
    device = resolve_training_device()
    print(f"device={device.type}")
    metadata_extra: dict[str, object] = {
        "patch_root": str(patch_root),
        "patch_roots": [str(root) for root in patch_roots],
        "sample_count": len(samples),
        "mode": "five_band_patch_train",
        "in_channels": in_channels,
        "required_positive_heads": list(required_positive_heads),
        "include_empty_scene_ids": list(include_empty_scene_ids),
        "sampling_strategy": args.sampling_strategy,
        "competition_weight": args.competition_weight,
        "paint_positive_weight": args.paint_positive_weight,
        "pollution_positive_weight": args.pollution_positive_weight,
        "aging_positive_weight": args.aging_positive_weight,
        "overlap_penalty_weight": args.overlap_penalty_weight,
        "spectral_loss_weight": args.spectral_loss_weight,
        "disease_spectral_weight": args.disease_spectral_weight,
        "aging_exclusivity_weight": args.aging_exclusivity_weight,
        "use_spectral_se": args.use_spectral_se,
        "device": device.type,
        "python_executable": sys.executable,
        "resume_checkpoint_path": str(resume_checkpoint) if resume_checkpoint else None,
        "sample_record_path": str(sample_record_path),
        "pigment_class_names": list(pigment_class_names),
        "pigment_loss_weight": args.pigment_loss_weight,
        "learning_rate": args.learning_rate if args.learning_rate is not None else (
            5e-5 if args.background4_v3_agingfix else (1e-4 if args.background4_v3 else 1e-3)
        ),
        "pigment_label_mode": args.pigment_label_mode,
        "pigment_masked_pooling": False,
        "pigment_pixelwise": is_background4,
        "version_id": version_id if is_background4 else None,
        "rotation_augmentation": is_background4,
        "background_conditioning": is_background_conditioned,
        "background_roles": list(BACKGROUND4_ROLES) if is_background_conditioned else None,
        "pure_background_multiplier": 2.5 if is_background_conditioned else 1.0,
        "sampling_groups": {
            name: {"scene_ids": sorted(scene_ids), "target_fraction": fraction}
            for name, (scene_ids, fraction) in sampling_groups.items()
        },
        "training_manifest_path": str(args.training_manifest) if args.training_manifest else None,
        "training_manifest_sha256": manifest_checksum,
        "checkpoint_every": args.checkpoint_every,
    }

    model = build_multitask_model(
        variant=args.model_variant,
        encoder_name="resnet18",
        in_channels=in_channels,
        head_names=("paint", "pollution", "aging"),
        use_spectral_se=args.use_spectral_se,
        pigment_class_count=len(pigment_class_names),
        pigment_masked_pooling=False,
        pigment_pixelwise=is_background4,
    ).to(device)

    if aux_checkpoint and aux_checkpoint.exists():
        summary = load_rgb_auxiliary_checkpoint(model, aux_checkpoint)
        print(
            "Loaded RGB auxiliary checkpoint "
            f"({summary['loaded_keys']} keys, skipped {summary['skipped_keys']} keys)"
        )

    if resume_checkpoint:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"resume_checkpoint={resume_checkpoint}")
        print(f"resume_missing_keys={len(load_result.missing_keys)}")
        print(f"resume_unexpected_keys={len(load_result.unexpected_keys)}")

    learning_rate = args.learning_rate if args.learning_rate is not None else (
        5e-5 if args.background4_v3_agingfix else (1e-4 if args.background4_v3 else 1e-3)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_loss = 0.0
    loss_history: list[float] = []
    tracked_task_names = ["paint", "pollution", "aging"]
    if pigment_class_names:
        tracked_task_names.append("pigment")
    task_loss_history = build_task_loss_history(tracked_task_names)
    task_weight_history: list[dict[str, float]] = []
    experiment_dir = resolve_experiment_dir(
        PROJECT_ROOT,
        args.model_variant,
        args.epochs,
        args.output_dir,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        if args.loss_weighting == "dwa":
            task_weights = compute_dwa_weights(
                {name: task_loss_history[name] for name in tracked_task_names},
                temperature=args.dwa_temperature,
            )
        else:
            task_weights = {name: 1.0 for name in tracked_task_names}
        if "pigment" in task_weights:
            task_weights["pigment"] *= args.pigment_loss_weight
        task_weight_history.append(task_weights.copy())
        last_loss, task_means = run_vnir_training_epoch(
            model,
            dataloader,
            optimizer,
            device,
            task_weights=task_weights,
            competition_weight=args.competition_weight,
            paint_positive_weight=args.paint_positive_weight,
            pollution_positive_weight=args.pollution_positive_weight,
            aging_positive_weight=args.aging_positive_weight,
            overlap_penalty_weight=args.overlap_penalty_weight,
            spectral_loss_weight=args.spectral_loss_weight,
            disease_spectral_weight=args.disease_spectral_weight,
            aging_exclusivity_weight=args.aging_exclusivity_weight,
        )
        loss_history.append(last_loss)
        for name, value in task_means.items():
            task_loss_history[name].append(value)
        print(f"epoch={epoch + 1} loss={last_loss:.6f}")
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            epoch_checkpoint_path = experiment_dir / f"{version_id}_epoch_{epoch + 1:03d}.pt"
            torch.save(
                build_checkpoint_metadata(
                    model_variant=args.model_variant,
                    loss_weighting=args.loss_weighting,
                    dwa_temperature=args.dwa_temperature,
                    task_loss_history=task_loss_history,
                    task_weight_history=task_weight_history,
                    model_state_dict=model.state_dict(),
                    epochs=args.epochs,
                    epochs_completed=epoch + 1,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    final_loss=last_loss,
                    loss_history=loss_history,
                    **metadata_extra,
                ),
                epoch_checkpoint_path,
            )

    checkpoint_path = experiment_dir / args.checkpoint_name
    log_path = experiment_dir / "training_log.txt"
    log_lines = [
        f"model_variant={args.model_variant}",
        f"loss_weighting={args.loss_weighting}",
        f"dwa_temperature={args.dwa_temperature}",
        f"patch_root={metadata_extra['patch_root']}",
        f"sample_count={metadata_extra['sample_count']}",
        f"mode={metadata_extra['mode']}",
        f"in_channels={metadata_extra['in_channels']}",
        "required_positive_heads=" + ",".join(required_positive_heads),
        "include_empty_scene_ids=" + ",".join(include_empty_scene_ids),
        f"sampling_strategy={metadata_extra['sampling_strategy']}",
        f"competition_weight={metadata_extra['competition_weight']}",
        f"paint_positive_weight={metadata_extra['paint_positive_weight']}",
        f"pollution_positive_weight={metadata_extra['pollution_positive_weight']}",
        f"aging_positive_weight={metadata_extra['aging_positive_weight']}",
        f"overlap_penalty_weight={metadata_extra['overlap_penalty_weight']}",
        f"spectral_loss_weight={metadata_extra['spectral_loss_weight']}",
        f"disease_spectral_weight={metadata_extra['disease_spectral_weight']}",
        f"aging_exclusivity_weight={metadata_extra['aging_exclusivity_weight']}",
        f"use_spectral_se={metadata_extra['use_spectral_se']}",
        f"device={metadata_extra['device']}",
        f"python_executable={metadata_extra['python_executable']}",
        f"resume_checkpoint_path={metadata_extra['resume_checkpoint_path'] or 'none'}",
        f"sample_record_path={metadata_extra['sample_record_path']}",
        "pigment_class_names=" + ",".join(metadata_extra['pigment_class_names']),
        f"pigment_loss_weight={metadata_extra['pigment_loss_weight']}",
    ]
    for index, loss in enumerate(loss_history):
        log_lines.extend(
            [
                f"epoch={index + 1} loss={loss:.6f}",
                f"epoch={index + 1} paint_loss={task_loss_history['paint'][index]:.6f}",
                f"epoch={index + 1} pollution_loss={task_loss_history['pollution'][index]:.6f}",
                f"epoch={index + 1} aging_loss={task_loss_history['aging'][index]:.6f}",
                *([f"epoch={index + 1} pigment_loss={task_loss_history['pigment'][index]:.6f}"] if "pigment" in task_loss_history else []),
                f"epoch={index + 1} paint_pollution_competition_loss={task_loss_history['paint_pollution_competition'][index]:.6f}",
                f"epoch={index + 1} paint_pollution_overlap_loss={task_loss_history['paint_pollution_overlap'][index]:.6f}",
                f"epoch={index + 1} spectral_guidance_loss={task_loss_history['spectral_guidance'][index]:.6f}",
                f"epoch={index + 1} aging_paint_edge_loss={task_loss_history['aging_paint_edge'][index]:.6f}",
                f"epoch={index + 1} disease_spectral_prior_loss={task_loss_history['disease_spectral_prior'][index]:.6f}",
                f"epoch={index + 1} aging_exclusivity_loss={task_loss_history['aging_exclusivity'][index]:.6f}",
                f"epoch={index + 1} paint_weight={task_weight_history[index]['paint']:.6f}",
                f"epoch={index + 1} pollution_weight={task_weight_history[index]['pollution']:.6f}",
                f"epoch={index + 1} aging_weight={task_weight_history[index]['aging']:.6f}",
                *([f"epoch={index + 1} pigment_weight={task_weight_history[index]['pigment']:.6f}"] if "pigment" in task_weight_history[index] else []),
            ]
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    torch.save(
        build_checkpoint_metadata(
            model_variant=args.model_variant,
            loss_weighting=args.loss_weighting,
            dwa_temperature=args.dwa_temperature,
            task_loss_history=task_loss_history,
            task_weight_history=task_weight_history,
            model_state_dict=model.state_dict(),
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            final_loss=last_loss,
            loss_history=loss_history,
            **metadata_extra,
        ),
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"training_log={log_path}")


if __name__ == "__main__":
    main()
