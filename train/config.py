from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_zip: str
    modality: str
    model_name: str
    encoder_name: str
    task_mode: str
    head_names: tuple[str, str, str]
    head_output_channels: dict[str, int]
    patch_size: int


def create_experiment_config(dataset_zip: str) -> ExperimentConfig:
    head_names = ("paint", "pollution", "aging")
    return ExperimentConfig(
        dataset_zip=dataset_zip,
        modality="VNIR",
        model_name="UnetPlusPlus",
        encoder_name="resnet18",
        task_mode="multitask",
        head_names=head_names,
        head_output_channels={name: 1 for name in head_names},
        patch_size=256,
    )
