import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_dinov2_channels(model, model_name):
    if hasattr(model, "feature_info"):
        channels = model.feature_info.channels()
        if channels:
            return channels[-1]

    if "large" in model_name:
        return 1024
    if "base" in model_name:
        return 768
    if "small" in model_name:
        return 384
    if "giant" in model_name or "g" in model_name:
        return 1536

    raise ValueError(f"Cannot infer DINOv2 feature channels for {model_name}")


class CGDF(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.dynamic_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1, bias=False),
            nn.Softmax(dim=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, S, T):
        S_calib = S * self.channel_gate(T) + S
        T_calib = T * self.spatial_gate(S) + T

        cat_pool = torch.cat([S_calib, T_calib], dim=1)
        weights = self.dynamic_weight(cat_pool)
        w_s, w_t = weights[:, 0:1], weights[:, 1:2]
        fused = w_s * S_calib + w_t * T_calib

        out = self.refine(fused) + S
        return out


class DinoConcatFusion(nn.Module):

    def __init__(
        self,
        rgb_channels=(64, 128, 320, 512),
        model_name="vit_base_patch14_dinov2",
        input_size=448,
        checkpoint_path=None,
        pretrained=None,
        freeze=True,
    ):
        super().__init__()
        if len(rgb_channels) != 4:
            raise ValueError("rgb_channels must contain four channel sizes")

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "DINOv2 fusion requires timm. Install a recent timm version "
                "that supports DINOv2 models."
            ) from exc

        if pretrained is None:
            pretrained = checkpoint_path is not None

        pretrained_cfg_overlay = None
        if checkpoint_path:
            pretrained_cfg_overlay = {"file": checkpoint_path}

        create_kwargs = dict(
            features_only=True,
            img_size=(input_size, input_size),
            pretrained=pretrained,
        )
        if pretrained_cfg_overlay is not None:
            create_kwargs["pretrained_cfg_overlay"] = pretrained_cfg_overlay

        self.dino = timm.create_model(model_name, **create_kwargs)
        self.input_size = input_size
        self.freeze = freeze

        if freeze:
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False

        dino_channels = _infer_dinov2_channels(self.dino, model_name)

        self.align = nn.ModuleList(
            [nn.Conv2d(dino_channels, channel, kernel_size=1)
             for channel in rgb_channels]
        )
        self.fusers = nn.ModuleList(
            [CGDF(channel) for channel in rgb_channels]
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self

    def forward(self, rgb_features, image):
        if len(rgb_features) != 4:
            raise ValueError("rgb_features must contain four feature maps")

        with torch.no_grad():
            dino_input = F.interpolate(
                image,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            dino_feature = self.dino(dino_input)[-1]

        if dino_feature.ndim != 4:
            raise ValueError(
                f"DINOv2 feature must be 4D, got {tuple(dino_feature.shape)}"
            )

        if dino_feature.shape[1] != self.align[0].in_channels:
            dino_feature = dino_feature.permute(0, 3, 1, 2).contiguous()

        fused = []
        for rgb_feature, align, fuser in zip(
            rgb_features,
            self.align,
            self.fusers,
        ):
            dino_aligned = align(dino_feature)
            dino_aligned = F.interpolate(
                dino_aligned,
                size=rgb_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused.append(fuser(rgb_feature, dino_aligned))

        return tuple(fused)
