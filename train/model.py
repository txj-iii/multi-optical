from __future__ import annotations

import torch
from torch import nn
import segmentation_models_pytorch as smp

MODEL_VARIANTS = ("baseline", "attention", "task_specific")


def _build_backbone(encoder_name: str, in_channels: int, shared_channels: int) -> nn.Module:
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=shared_channels,
    )


def _build_heads(head_names: tuple[str, ...], shared_channels: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {name: nn.Conv2d(shared_channels, 1, kernel_size=1) for name in head_names}
    )


def _build_pigment_head(shared_channels: int, pigment_class_count: int | None) -> nn.Module | None:
    if not pigment_class_count:
        return None
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(shared_channels, pigment_class_count),
    )


def _build_pixel_pigment_head(shared_channels: int, pigment_class_count: int | None) -> nn.Module | None:
    """Four-class per-pixel pigment logits for background4_v1."""
    if not pigment_class_count:
        return None
    return nn.Conv2d(shared_channels, pigment_class_count, kernel_size=1)


class MaskedPigmentHead(nn.Module):
    """Classify only features inside a paint mask.

    The mask is detached deliberately: pigment supervision must not teach the
    paint head to enlarge its foreground area.
    """
    def __init__(self, channels: int, class_count: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(channels, class_count)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != features.shape[-2:]:
            mask = torch.nn.functional.interpolate(mask, size=features.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.detach().clamp(0, 1)
        denominator = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = (features * mask).sum(dim=(2, 3)) / denominator
        return self.classifier(pooled)


class SpectralSEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 3) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.excitation(self.pool(x))
        return x * weights


class TaskAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class TaskRefinementBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MultiTaskUnetPlusPlus(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        head_names: tuple[str, ...],
        shared_channels: int = 16,
        use_spectral_se: bool = False,
        pigment_class_count: int | None = None,
        pigment_masked_pooling: bool = False,
        pigment_pixelwise: bool = False,
    ) -> None:
        super().__init__()
        self.head_names = head_names
        self.input_attention = SpectralSEBlock(in_channels) if use_spectral_se else nn.Identity()
        self.backbone = _build_backbone(encoder_name, in_channels, shared_channels)
        self.heads = _build_heads(head_names, shared_channels)
        self.pigment_masked_pooling = pigment_masked_pooling
        self.pigment_pixelwise = pigment_pixelwise
        self.pigment_head = (_build_pixel_pigment_head(shared_channels, pigment_class_count) if pigment_pixelwise else (MaskedPigmentHead(shared_channels, pigment_class_count) if pigment_masked_pooling and pigment_class_count else _build_pigment_head(shared_channels, pigment_class_count)))

    def forward(self, x: torch.Tensor, pigment_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.input_attention(x)
        shared_features = self.backbone(x)
        outputs = {name: head(shared_features) for name, head in self.heads.items()}
        if self.pigment_head is not None:
            mask = pigment_mask if pigment_mask is not None else torch.sigmoid(outputs["paint"])
            outputs["pigment"] = self.pigment_head(shared_features, mask) if self.pigment_masked_pooling else self.pigment_head(shared_features)
        return outputs


class MultiTaskAttentionUnetPlusPlus(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        head_names: tuple[str, ...],
        shared_channels: int = 16,
        use_spectral_se: bool = False,
        pigment_class_count: int | None = None,
        pigment_masked_pooling: bool = False,
        pigment_pixelwise: bool = False,
    ) -> None:
        super().__init__()
        self.head_names = head_names
        self.input_attention = SpectralSEBlock(in_channels) if use_spectral_se else nn.Identity()
        self.backbone = _build_backbone(encoder_name, in_channels, shared_channels)
        self.attentions = nn.ModuleDict(
            {name: TaskAttention(shared_channels) for name in head_names}
        )
        self.heads = _build_heads(head_names, shared_channels)
        self.pigment_attention = TaskAttention(shared_channels) if pigment_class_count else None
        self.pigment_masked_pooling = pigment_masked_pooling
        self.pigment_pixelwise = pigment_pixelwise
        self.pigment_head = (_build_pixel_pigment_head(shared_channels, pigment_class_count) if pigment_pixelwise else (MaskedPigmentHead(shared_channels, pigment_class_count) if pigment_masked_pooling and pigment_class_count else _build_pigment_head(shared_channels, pigment_class_count)))

    def forward(self, x: torch.Tensor, pigment_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.input_attention(x)
        shared_features = self.backbone(x)
        outputs: dict[str, torch.Tensor] = {}
        for name in self.head_names:
            task_features = self.attentions[name](shared_features)
            outputs[name] = self.heads[name](task_features)
        if self.pigment_head is not None and self.pigment_attention is not None:
            features = self.pigment_attention(shared_features)
            mask = pigment_mask if pigment_mask is not None else torch.sigmoid(outputs["paint"])
            outputs["pigment"] = self.pigment_head(features, mask) if self.pigment_masked_pooling else self.pigment_head(features)
        return outputs


class MultiTaskSpecificUnetPlusPlus(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        head_names: tuple[str, ...],
        shared_channels: int = 16,
        use_spectral_se: bool = False,
        pigment_class_count: int | None = None,
        pigment_masked_pooling: bool = False,
        pigment_pixelwise: bool = False,
    ) -> None:
        super().__init__()
        self.head_names = head_names
        self.input_attention = SpectralSEBlock(in_channels) if use_spectral_se else nn.Identity()
        self.backbone = _build_backbone(encoder_name, in_channels, shared_channels)
        self.refinement_blocks = nn.ModuleDict(
            {name: TaskRefinementBlock(shared_channels) for name in head_names}
        )
        self.heads = _build_heads(head_names, shared_channels)
        self.pigment_refinement = TaskRefinementBlock(shared_channels) if pigment_class_count else None
        self.pigment_masked_pooling = pigment_masked_pooling
        self.pigment_pixelwise = pigment_pixelwise
        self.pigment_head = (_build_pixel_pigment_head(shared_channels, pigment_class_count) if pigment_pixelwise else (MaskedPigmentHead(shared_channels, pigment_class_count) if pigment_masked_pooling and pigment_class_count else _build_pigment_head(shared_channels, pigment_class_count)))

    def forward(self, x: torch.Tensor, pigment_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.input_attention(x)
        shared_features = self.backbone(x)
        outputs: dict[str, torch.Tensor] = {}
        for name in self.head_names:
            refined_features = self.refinement_blocks[name](shared_features)
            outputs[name] = self.heads[name](refined_features)
        if self.pigment_head is not None and self.pigment_refinement is not None:
            features = self.pigment_refinement(shared_features)
            mask = pigment_mask if pigment_mask is not None else torch.sigmoid(outputs["paint"])
            outputs["pigment"] = self.pigment_head(features, mask) if self.pigment_masked_pooling else self.pigment_head(features)
        return outputs


def build_multitask_model(
    variant: str,
    encoder_name: str,
    in_channels: int,
    head_names: tuple[str, ...],
    shared_channels: int = 16,
    use_spectral_se: bool = False,
    pigment_class_count: int | None = None,
    pigment_masked_pooling: bool = False,
    pigment_pixelwise: bool = False,
) -> nn.Module:
    if variant == "baseline":
        return MultiTaskUnetPlusPlus(
            encoder_name=encoder_name,
            in_channels=in_channels,
            head_names=head_names,
            shared_channels=shared_channels,
            use_spectral_se=use_spectral_se,
            pigment_class_count=pigment_class_count,
            pigment_masked_pooling=pigment_masked_pooling,
            pigment_pixelwise=pigment_pixelwise,
        )
    if variant == "attention":
        return MultiTaskAttentionUnetPlusPlus(
            encoder_name=encoder_name,
            in_channels=in_channels,
            head_names=head_names,
            shared_channels=shared_channels,
            use_spectral_se=use_spectral_se,
            pigment_class_count=pigment_class_count,
            pigment_masked_pooling=pigment_masked_pooling,
            pigment_pixelwise=pigment_pixelwise,
        )
    if variant == "task_specific":
        return MultiTaskSpecificUnetPlusPlus(
            encoder_name=encoder_name,
            in_channels=in_channels,
            head_names=head_names,
            shared_channels=shared_channels,
            use_spectral_se=use_spectral_se,
            pigment_class_count=pigment_class_count,
            pigment_masked_pooling=pigment_masked_pooling,
            pigment_pixelwise=pigment_pixelwise,
        )
    raise ValueError(f"Unsupported model variant: {variant}")
