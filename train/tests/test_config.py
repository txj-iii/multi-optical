from train.config import create_experiment_config


def test_create_experiment_config_defaults_to_vnir_multitask_unetplusplus_resnet18() -> None:
    config = create_experiment_config(dataset_zip="train/five_band_patches/train")

    assert config.dataset_zip == "train/five_band_patches/train"
    assert config.modality == "VNIR"
    assert config.model_name == "UnetPlusPlus"
    assert config.encoder_name == "resnet18"
    assert config.task_mode == "multitask"
    assert config.patch_size == 256
    assert config.head_names == ("paint", "pollution", "aging")
    assert config.head_output_channels == {"paint": 1, "pollution": 1, "aging": 1}
