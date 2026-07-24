from train import run_training


def test_default_patch_root_matches_current_validation_patch_root() -> None:
    assert str(run_training.DEFAULT_PATCH_ROOT_REL).replace('\\', '/') == 'train/five_band_patches/train_light_v10_balanced_4849_pollution4447'
