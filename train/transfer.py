from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_rgb_auxiliary_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, int | str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()

    filtered_state = {}
    skipped_keys = 0
    for key, value in source_state.items():
        if key not in target_state:
            skipped_keys += 1
            continue
        if target_state[key].shape != value.shape:
            skipped_keys += 1
            continue
        filtered_state[key] = value

    missing_before = len(target_state) - len(filtered_state)
    model.load_state_dict(filtered_state, strict=False)

    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_keys": len(filtered_state),
        "skipped_keys": skipped_keys + missing_before - max(missing_before - skipped_keys, 0),
    }
