from train.aging_only_finetune import _freeze_non_aging_head_parameters, parse_args
from train.model import build_multitask_model


def test_aging_only_finetune_defaults_to_focus_aging_sampling() -> None:
    args = parse_args(["--base-checkpoint-path", "C:/demo/base.pt"])

    assert args.sampling_strategy == "aging_focus"
    assert args.focus_scene_ids == ["SAMPLE_019", "SAMPLE_024", "SAMPLE_025", "SAMPLE_026", "SAMPLE_027"]
    assert args.focus_positive_repeats == 4
    assert args.focus_recall_threshold == 0.5


def test_aging_only_finetune_accepts_focus_sampling_overrides() -> None:
    args = parse_args(
        [
            "--base-checkpoint-path",
            "C:/demo/base.pt",
            "--focus-scene-ids",
            "SAMPLE_024",
            "SAMPLE_025",
            "--focus-positive-repeats",
            "6",
            "--focus-scene-multiplier",
            "7.5",
            "--aging-positive-sample-multiplier",
            "3.0",
            "--focus-recall-threshold",
            "0.35",
        ]
    )

    assert args.focus_scene_ids == ["SAMPLE_024", "SAMPLE_025"]
    assert args.focus_positive_repeats == 6
    assert args.focus_scene_multiplier == 7.5
    assert args.aging_positive_sample_multiplier == 3.0
    assert args.focus_recall_threshold == 0.35




def test_aging_only_finetune_accepts_paint_edge_suppression_overrides() -> None:
    args = parse_args(
        [
            "--base-checkpoint-path",
            "C:/demo/base.pt",
            "--paint-edge-negative-weight",
            "0.7",
            "--paint-edge-width",
            "3",
        ]
    )

    assert args.paint_edge_negative_weight == 0.7
    assert args.paint_edge_width == 3
def test_freeze_non_aging_head_parameters_keeps_only_aging_branch_trainable() -> None:
    model = build_multitask_model(
        variant="task_specific",
        encoder_name="resnet18",
        in_channels=15,
        head_names=("paint", "pollution", "aging"),
        use_spectral_se=True,
    )

    _freeze_non_aging_head_parameters(model)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all(
        name.startswith("heads.aging.") or name.startswith("refinement_blocks.aging.")
        for name in trainable
    )
    assert not any(name.startswith("backbone.") for name in trainable)
    assert not any(name.startswith("input_attention.") for name in trainable)


def test_freeze_non_aging_head_parameters_keeps_only_aging_branch_trainable_with_pigment_head() -> None:
    model = build_multitask_model(
        variant="task_specific",
        encoder_name="resnet18",
        in_channels=15,
        head_names=("paint", "pollution", "aging"),
        use_spectral_se=True,
        pigment_class_count=7,
    )

    _freeze_non_aging_head_parameters(model)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all(
        name.startswith("heads.aging.") or name.startswith("refinement_blocks.aging.") or name.startswith("attentions.aging.")
        for name in trainable
    )
    assert not any(name.startswith("pigment_") for name in trainable)
