from pathlib import Path

import torch

from train.model import MultiTaskUnetPlusPlus
from train.transfer import load_rgb_auxiliary_checkpoint


def _write_aux_checkpoint(checkpoint_path: Path) -> None:
    source_model = MultiTaskUnetPlusPlus(
        encoder_name="resnet18",
        in_channels=3,
        head_names=("damage",),
    )
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)


def test_load_rgb_auxiliary_checkpoint_returns_summary(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "muraldh_pretrain.pt"
    _write_aux_checkpoint(checkpoint_path)
    model = MultiTaskUnetPlusPlus(
        encoder_name="resnet18",
        in_channels=5,
        head_names=("paint", "pollution", "aging"),
    )

    summary = load_rgb_auxiliary_checkpoint(model, checkpoint_path)

    assert summary["checkpoint_path"] == str(checkpoint_path)
    assert summary["loaded_keys"] > 0
    assert summary["skipped_keys"] >= 1


def test_transfer_loader_keeps_multichannel_input_stem_shape(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "muraldh_pretrain.pt"
    _write_aux_checkpoint(checkpoint_path)
    model = MultiTaskUnetPlusPlus(
        encoder_name="resnet18",
        in_channels=5,
        head_names=("paint", "pollution", "aging"),
    )

    before_shape = model.backbone.encoder.conv1.weight.shape
    load_rgb_auxiliary_checkpoint(model, checkpoint_path)
    after_shape = model.backbone.encoder.conv1.weight.shape

    assert before_shape == torch.Size([64, 5, 7, 7])
    assert after_shape == before_shape
