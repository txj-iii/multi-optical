from pathlib import Path

from train.run_training import build_task_loss_history, resolve_patch_root


def test_resolve_patch_root_defaults_to_train_light_v5_focus323335() -> None:
    project_root = Path('D:/multi-optical')

    resolved = resolve_patch_root(project_root, None)

    assert resolved == project_root / 'train' / 'five_band_patches' / 'train_light_v5_focus323335'


def test_resolve_patch_root_keeps_explicit_override() -> None:
    project_root = Path('D:/multi-optical')
    explicit = 'D:/multi-optical/train/five_band_patches/train'

    resolved = resolve_patch_root(project_root, explicit)

    assert resolved == Path(explicit)


def test_build_task_loss_history_includes_auxiliary_losses() -> None:
    history = build_task_loss_history(["paint", "pollution", "aging", "pigment"])

    assert set(history) >= {
        "paint",
        "pollution",
        "aging",
        "pigment",
        "paint_pollution_competition",
        "paint_pollution_overlap",
        "spectral_guidance",
        "aging_paint_edge",
    }
    assert all(values == [] for values in history.values())
