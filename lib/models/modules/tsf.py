from torch import nn

from .dinov2_fusion import DinoConcatFusion
from .depth_fusion import GDF


class TSF(nn.Module):

    def __init__(
        self,
        rgb_channels=(64, 128, 320, 512),
        use_dinov2=True,
        dinov2_model='vit_large_patch14_dinov2',
        dinov2_path='./pretrained/dinov2_large.safetensors',
        dinov2_input_size=448,
        freeze_dinov2=True,
    ):
        super().__init__()
        self.cgdf = (
            DinoConcatFusion(
                rgb_channels=rgb_channels,
                model_name=dinov2_model,
                input_size=dinov2_input_size,
                checkpoint_path=dinov2_path,
                freeze=freeze_dinov2,
            )
            if use_dinov2 else None
        )
        self.gdf = GDF(rgb_channels=rgb_channels)

    def forward(self, rgb_features, image, depth):
        if self.cgdf is not None:
            rgb_features = self.cgdf(rgb_features, image)
        return self.gdf(*rgb_features, depth)
