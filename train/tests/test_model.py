import torch

from train.config import create_experiment_config
from train.model import MultiTaskUnetPlusPlus, SpectralSEBlock, build_multitask_model


def test_multitask_unetplusplus_emits_three_binary_heads() -> None:
    config = create_experiment_config(dataset_zip="train/five_band_patches/train")
    model = MultiTaskUnetPlusPlus(
        encoder_name=config.encoder_name,
        in_channels=186,
        head_names=config.head_names,
    )

    batch = torch.randn(2, 186, 64, 64)
    outputs = model(batch)

    assert set(outputs) == {"paint", "pollution", "aging"}
    assert outputs["paint"].shape == (2, 1, 64, 64)
    assert outputs["pollution"].shape == (2, 1, 64, 64)
    assert outputs["aging"].shape == (2, 1, 64, 64)


def test_attention_multitask_model_emits_three_binary_heads() -> None:
    model = build_multitask_model(
        variant="attention",
        encoder_name="resnet18",
        in_channels=3,
        head_names=("paint", "pollution", "aging"),
    )

    batch = torch.randn(2, 3, 64, 64)
    outputs = model(batch)

    assert set(outputs) == {"paint", "pollution", "aging"}
    assert outputs["paint"].shape == (2, 1, 64, 64)
    assert outputs["pollution"].shape == (2, 1, 64, 64)
    assert outputs["aging"].shape == (2, 1, 64, 64)


def test_task_specific_multitask_model_emits_three_binary_heads() -> None:
    model = build_multitask_model(
        variant="task_specific",
        encoder_name="resnet18",
        in_channels=3,
        head_names=("paint", "pollution", "aging"),
    )

    batch = torch.randn(2, 3, 64, 64)
    outputs = model(batch)

    assert set(outputs) == {"paint", "pollution", "aging"}
    assert outputs["paint"].shape == (2, 1, 64, 64)
    assert outputs["pollution"].shape == (2, 1, 64, 64)
    assert outputs["aging"].shape == (2, 1, 64, 64)


def test_spectral_se_block_keeps_shape_and_learns_channel_weights() -> None:
    block = SpectralSEBlock(channels=9, reduction=3)
    batch = torch.randn(2, 9, 16, 16)

    output = block(batch)

    assert output.shape == batch.shape
    assert any(parameter.requires_grad for parameter in block.parameters())


def test_task_specific_model_can_enable_spectral_se_input_attention() -> None:
    model = build_multitask_model(
        variant="task_specific",
        encoder_name="resnet18",
        in_channels=15,
        head_names=("paint", "pollution", "aging"),
        use_spectral_se=True,
    )

    batch = torch.randn(2, 15, 64, 64)
    outputs = model(batch)

    assert set(outputs) == {"paint", "pollution", "aging"}
    assert model.input_attention.__class__.__name__ == "SpectralSEBlock"


def test_task_specific_model_can_emit_pigment_aux_logits() -> None:
    model = build_multitask_model(
        variant="task_specific",
        encoder_name="resnet18",
        in_channels=15,
        head_names=("paint", "pollution", "aging"),
        pigment_class_count=7,
        use_spectral_se=True,
    )

    batch = torch.randn(2, 15, 64, 64)
    outputs = model(batch)

    assert set(outputs) == {"paint", "pollution", "aging", "pigment"}
    assert outputs["paint"].shape == (2, 1, 64, 64)
    assert outputs["pollution"].shape == (2, 1, 64, 64)
    assert outputs["aging"].shape == (2, 1, 64, 64)
    assert outputs["pigment"].shape == (2, 7)
