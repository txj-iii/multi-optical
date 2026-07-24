from __future__ import annotations

import numpy as np

RATIO_EPS = 1e-6


def augment_five_band_cube_with_spectral_features(five_band: np.ndarray) -> np.ndarray:
    if five_band.ndim != 3 or five_band.shape[2] != 5:
        raise ValueError(f"Expected five-band cube with shape HxWx5, got {five_band.shape}.")

    cube = five_band.astype(np.float32, copy=False)
    diff_550_450 = (cube[:, :, 1] - cube[:, :, 0])[:, :, None]
    diff_600_550 = (cube[:, :, 2] - cube[:, :, 1])[:, :, None]
    diff_650_600 = (cube[:, :, 3] - cube[:, :, 2])[:, :, None]
    diff_700_650 = (cube[:, :, 4] - cube[:, :, 3])[:, :, None]
    ratio_550_450 = (cube[:, :, 1] / (cube[:, :, 0] + RATIO_EPS))[:, :, None]
    ratio_600_550 = (cube[:, :, 2] / (cube[:, :, 1] + RATIO_EPS))[:, :, None]
    ratio_650_600 = (cube[:, :, 3] / (cube[:, :, 2] + RATIO_EPS))[:, :, None]
    ratio_700_650 = (cube[:, :, 4] / (cube[:, :, 3] + RATIO_EPS))[:, :, None]
    nd_700_450 = ((cube[:, :, 4] - cube[:, :, 0]) / (cube[:, :, 4] + cube[:, :, 0] + RATIO_EPS))[:, :, None]
    nd_650_550 = ((cube[:, :, 3] - cube[:, :, 1]) / (cube[:, :, 3] + cube[:, :, 1] + RATIO_EPS))[:, :, None]
    return np.concatenate(
        [
            cube,
            diff_550_450.astype(np.float32, copy=False),
            diff_600_550.astype(np.float32, copy=False),
            diff_650_600.astype(np.float32, copy=False),
            diff_700_650.astype(np.float32, copy=False),
            ratio_550_450.astype(np.float32, copy=False),
            ratio_600_550.astype(np.float32, copy=False),
            ratio_650_600.astype(np.float32, copy=False),
            ratio_700_650.astype(np.float32, copy=False),
            nd_700_450.astype(np.float32, copy=False),
            nd_650_550.astype(np.float32, copy=False),
        ],
        axis=2,
    )


def infer_augmented_five_band_channel_count(raw_channel_count: int) -> int:
    if raw_channel_count != 5:
        raise ValueError(f"Expected raw five-band input to contain 5 channels, got {raw_channel_count}.")
    return 15
