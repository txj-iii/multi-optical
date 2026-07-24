from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.model import build_multitask_model
from train.run_training import resolve_patch_root
from train.vnir_train import (
    build_six_band_dataloader,
    collect_six_band_patch_samples,
    compute_aging_paint_edge_suppression_loss,
    compute_binary_segmentation_loss,
    oversample_focus_aging_samples,
    sample_has_positive_aging,
)


def resolve_training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune only the aging branch from an existing checkpoint.")
    parser.add_argument("--base-checkpoint-path", type=str, required=True, help="Existing multitask checkpoint path.")
    parser.add_argument(
        "--patch-root",
        type=str,
        default=None,
        help="Five-band patch split root. Defaults to train/five_band_patches/train_light_v10_balanced_4849_pollution4447.",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Optional override for image size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for aging-only fine-tune.")
    parser.add_argument(
        "--sampling-strategy",
        choices=("shuffle", "aging_focus"),
        default="aging_focus",
        help="Patch sampling strategy for aging-only fine-tuning.",
    )
    parser.add_argument(
        "--focus-scene-ids",
        nargs="*",
        default=["SAMPLE_019", "SAMPLE_024", "SAMPLE_025", "SAMPLE_026", "SAMPLE_027"],
        help="Scene IDs whose aging-positive patches should receive extra coverage.",
    )
    parser.add_argument(
        "--focus-positive-repeats",
        type=int,
        default=4,
        help="Deterministically repeat focus-scene aging-positive patches this many times per epoch.",
    )
    parser.add_argument(
        "--focus-scene-multiplier",
        type=float,
        default=5.0,
        help="Weighted-sampler multiplier for aging-positive patches from focus scenes.",
    )
    parser.add_argument(
        "--aging-positive-sample-multiplier",
        type=float,
        default=2.0,
        help="Weighted-sampler multiplier for any aging-positive patch.",
    )
    parser.add_argument(
        "--aging-positive-weight",
        type=float,
        default=4.0,
        help="Extra BCE positive weight applied only to aging pixels.",
    )
    parser.add_argument(
        "--focus-recall-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold used for focus-scene aging recall diagnostics.",
    )
    parser.add_argument(
        "--paint-edge-negative-weight",
        type=float,
        default=0.4,
        help="Extra suppression weight applied to non-aging pixels that sit on the paint edge band.",
    )
    parser.add_argument(
        "--paint-edge-width",
        type=int,
        default=2,
        help="Edge-band half-width in pixels used to mark paint borders as aging negatives.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args(argv)


def _freeze_non_aging_head_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        if (
            name.startswith("heads.aging.")
            or name.startswith("refinement_blocks.aging.")
            or name.startswith("attentions.aging.")
        ):
            parameter.requires_grad = True


def build_output_dir(project_root: Path, model_variant: str, epochs: int) -> Path:
    return (
        project_root
        / "train"
        / "experiments"
        / "five_band_train"
        / model_variant
        / f"aging_only_epochs_{epochs}"
    )


def compute_aging_recall(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    model.eval()
    true_positive_pixels = 0
    positive_pixels = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["aging_mask"].to(device)
            probabilities = torch.sigmoid(model(images)["aging"])
            predictions = probabilities >= threshold
            positives = targets > 0.5
            true_positive_pixels += int((predictions & positives).sum().item())
            positive_pixels += int(positives.sum().item())
    model.train()
    if positive_pixels == 0:
        return 1.0
    return true_positive_pixels / float(positive_pixels)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.base_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    patch_root = resolve_patch_root(PROJECT_ROOT, args.patch_root)
    samples = collect_six_band_patch_samples(patch_root, required_positive_heads=())
    if not samples:
        raise ValueError(f"No patches found under {patch_root}.")
    focus_scene_ids = tuple(args.focus_scene_ids or ())
    base_sample_count = len(samples)
    focus_eval_samples = [
        sample
        for sample in samples
        if sample.scene_id in set(focus_scene_ids) and sample_has_positive_aging(sample)
    ]
    samples = oversample_focus_aging_samples(
        samples,
        focus_scene_ids=focus_scene_ids,
        focus_positive_repeats=args.focus_positive_repeats,
    )

    image_size = int(args.image_size or checkpoint.get("image_size", 256))
    dataloader = build_six_band_dataloader(
        samples,
        image_size=image_size,
        batch_size=args.batch_size,
        sampling_strategy=args.sampling_strategy,
        focus_scene_ids=focus_scene_ids,
        focus_scene_multiplier=args.focus_scene_multiplier,
        aging_positive_multiplier=args.aging_positive_sample_multiplier,
    )
    focus_eval_dataloader = (
        build_six_band_dataloader(
            focus_eval_samples,
            image_size=image_size,
            batch_size=args.batch_size,
            sampling_strategy="shuffle",
        )
        if focus_eval_samples
        else None
    )

    model_variant = str(checkpoint.get("model_variant", "task_specific"))
    in_channels = int(checkpoint.get("in_channels", 15))
    pigment_class_names = tuple(checkpoint.get("pigment_class_names", ()))
    model = build_multitask_model(
        variant=model_variant,
        encoder_name="resnet18",
        in_channels=in_channels,
        head_names=("paint", "pollution", "aging"),
        use_spectral_se=bool(checkpoint.get("use_spectral_se", False)),
        pigment_class_count=len(pigment_class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _freeze_non_aging_head_parameters(model)
    device = resolve_training_device()
    print(f"device={device.type}")
    model = model.to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)

    loss_history: list[float] = []
    focus_recall_history: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_batches = 0
        for batch in dataloader:
            images = batch["image"].to(device)
            aging_mask = batch["aging_mask"].to(device)
            outputs = model(images)
            aging_loss = compute_binary_segmentation_loss(
                outputs["aging"],
                aging_mask,
                positive_weight=args.aging_positive_weight,
            )
            edge_loss = compute_aging_paint_edge_suppression_loss(
                outputs,
                {
                    "paint_mask": batch["paint_mask"].to(device),
                    "aging_mask": aging_mask,
                },
                edge_width=args.paint_edge_width,
            )
            total_step_loss = aging_loss + args.paint_edge_negative_weight * edge_loss
            optimizer.zero_grad()
            total_step_loss.backward()
            optimizer.step()
            total_loss += float(total_step_loss.item())
            total_batches += 1
        mean_loss = total_loss / max(total_batches, 1)
        loss_history.append(mean_loss)
        focus_recall = (
            compute_aging_recall(
                model,
                focus_eval_dataloader,
                device=device,
                threshold=args.focus_recall_threshold,
            )
            if focus_eval_dataloader is not None
            else 1.0
        )
        focus_recall_history.append(focus_recall)
        print(f"epoch={epoch + 1} aging_loss={mean_loss:.6f} focus_aging_recall={focus_recall:.6f}")

    output_dir = Path(args.output_dir) if args.output_dir else build_output_dir(PROJECT_ROOT, model_variant, args.epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "aging_only_finetune_latest.pt"
    output_log = output_dir / "training_log.txt"
    output_log.write_text(
        "\n".join(
            [
                f"base_checkpoint_path={checkpoint_path}",
                f"patch_root={patch_root}",
                f"base_sample_count={base_sample_count}",
                f"sample_count={len(samples)}",
                f"focus_eval_sample_count={len(focus_eval_samples)}",
                f"sampling_strategy={args.sampling_strategy}",
                f"focus_scene_ids={','.join(focus_scene_ids)}",
                f"focus_positive_repeats={args.focus_positive_repeats}",
                f"focus_scene_multiplier={args.focus_scene_multiplier}",
                f"aging_positive_sample_multiplier={args.aging_positive_sample_multiplier}",
                f"focus_recall_threshold={args.focus_recall_threshold}",
                f"paint_edge_negative_weight={args.paint_edge_negative_weight}",
                f"paint_edge_width={args.paint_edge_width}",
                f"in_channels={in_channels}",
                f"use_spectral_se={bool(checkpoint.get('use_spectral_se', False))}",
                f"device={device.type}",
                f"image_size={image_size}",
                f"lr={args.lr}",
                f"aging_positive_weight={args.aging_positive_weight}",
                *[f"epoch={index + 1} aging_loss={loss:.6f}" for index, loss in enumerate(loss_history)],
                *[
                    f"epoch={index + 1} focus_aging_recall={recall:.6f}"
                    for index, recall in enumerate(focus_recall_history)
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            **checkpoint,
            "model_state_dict": model.state_dict(),
            "aging_only_finetune": True,
            "base_checkpoint_path": str(checkpoint_path),
            "aging_positive_weight": args.aging_positive_weight,
            "sampling_strategy": args.sampling_strategy,
            "focus_scene_ids": list(focus_scene_ids),
            "focus_positive_repeats": args.focus_positive_repeats,
            "focus_scene_multiplier": args.focus_scene_multiplier,
            "aging_positive_sample_multiplier": args.aging_positive_sample_multiplier,
            "focus_recall_threshold": args.focus_recall_threshold,
            "paint_edge_negative_weight": args.paint_edge_negative_weight,
            "paint_edge_width": args.paint_edge_width,
            "aging_loss_history": loss_history,
            "focus_aging_recall_history": focus_recall_history,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": image_size,
        },
        output_checkpoint,
    )
    print(f"checkpoint={output_checkpoint}")
    print(f"training_log={output_log}")


if __name__ == "__main__":
    main()
