import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def resize_like(x, ref):
    return F.interpolate(
        x,
        size=ref.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


class DepthPyramidEncoder(nn.Module):

    def __init__(self, out_channels=(64, 128, 320, 512)):
        super().__init__()
        c1, c2, c3, c4 = out_channels

        self.stem = nn.Sequential(
            conv_bn_relu(1, c1 // 2, kernel_size=3, stride=2),
            conv_bn_relu(c1 // 2, c1, kernel_size=3, stride=2),
        )
        self.stage2 = nn.Sequential(
            conv_bn_relu(c1, c2, kernel_size=3, stride=2),
            conv_bn_relu(c2, c2, kernel_size=3, stride=1),
        )
        self.stage3 = nn.Sequential(
            conv_bn_relu(c2, c3, kernel_size=3, stride=2),
            conv_bn_relu(c3, c3, kernel_size=3, stride=1),
        )
        self.stage4 = nn.Sequential(
            conv_bn_relu(c3, c4, kernel_size=3, stride=2),
            conv_bn_relu(c4, c4, kernel_size=3, stride=1),
        )

    def forward(self, depth):
        if depth.ndim != 4 or depth.size(1) != 1:
            raise ValueError(
                "depth must have shape [B, 1, H, W], got {}".format(
                    tuple(depth.shape)
                )
            )

        depth = depth.float()
        d1 = self.stem(depth)
        d2 = self.stage2(d1)
        d3 = self.stage3(d2)
        d4 = self.stage4(d3)
        return d1, d2, d3, d4


class DepthGateFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine_depth = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feature, depth_feature, alpha):
        if depth_feature.shape[-2:] != rgb_feature.shape[-2:]:
            depth_feature = resize_like(depth_feature, rgb_feature)

        depth_feature = self.refine_depth(depth_feature)
        gate = self.gate(torch.cat((rgb_feature, depth_feature), dim=1))
        return rgb_feature + alpha * gate * depth_feature


class GDF(nn.Module):

    def __init__(self, rgb_channels=(64, 128, 320, 512)):
        super().__init__()
        if len(rgb_channels) != 4:
            raise ValueError("rgb_channels must contain four channel sizes")

        self.depth_encoder = DepthPyramidEncoder(rgb_channels)
        self.fusions = nn.ModuleList(
            [DepthGateFusion(channel) for channel in rgb_channels]
        )
        self.alpha = nn.Parameter(torch.full((4,), 0.1))

    def forward(self, x1, x2, x3, x4, depth):
        rgb_features = (x1, x2, x3, x4)
        depth_features = self.depth_encoder(depth)

        fused_features = tuple(
            fusion(rgb_feature, depth_feature, self.alpha[index])
            for index, (fusion, rgb_feature, depth_feature) in enumerate(
                zip(self.fusions, rgb_features, depth_features)
            )
        )
        return fused_features


PreReductionDepthFusion = GDF
