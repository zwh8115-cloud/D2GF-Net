import os

import torch
from torch import nn
from torch.nn import functional as F

from lib.RCAB import BasicConv2d, RFB_modified
from lib.models.backbones.pvtv2 import pvt_v2_b4
from lib.models.modules.tsf import TSF

class Reduction(nn.Module):

    def __init__(self, in_channel, out_channel, RFB=False):
        super().__init__()
        if RFB:
            self.reduce = nn.Sequential(
                RFB_modified(in_channel, out_channel),
            )
        else:
            self.reduce = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
            )

    def forward(self, x):
        return self.reduce(x)


class SVR(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.conv_sem = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.conv_vis = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel)
        )

    def forward(self, sem, vis):
        s = self.conv_sem(sem)
        v = self.conv_vis(vis)

        g = self.gate(torch.cat([s, v], dim=1))
        v_enh = v * g

        out = self.fuse(torch.cat([s, v_enh], dim=1))
        return out + v


class CCA(nn.Module):

    def __init__(self, channels=64):
        super().__init__()
        self.conv_high = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_cur = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f_cur, f_high):
        if f_cur.shape[2:] != f_high.shape[2:]:
            f_high = F.interpolate(f_high, size=f_cur.shape[2:], mode='bilinear', align_corners=False)

        f_high_proj = self.conv_high(f_high)
        f_cur_proj = self.conv_cur(f_cur)

        f_base = f_cur_proj + f_high_proj
        f_cons = f_cur_proj * f_high_proj

        mean_map = torch.mean(f_cons, dim=1, keepdim=True)
        max_map, _ = torch.max(f_cons, dim=1, keepdim=True)
        stat_map = torch.cat([mean_map, max_map], dim=1)
        m_s = self.spatial_gate(stat_map)

        m_c = self.channel_gate(f_cons)

        f_out = f_base + f_base * (m_s * m_c)
        return f_out


class Network(nn.Module):

    def __init__(
        self,
        channel=128,
        imagenet_pretrained=True,
        use_dinov2=True,
        dinov2_model='vit_large_patch14_dinov2',
        dinov2_path='./pretrained/dinov2_large.safetensors',
        dinov2_input_size=448,
        freeze_dinov2=True,
    ):
        super().__init__()

        self.bkbone = pvt_v2_b4()
        enc_channels = [64, 128, 320, 512]

        if imagenet_pretrained:
            pretrained_path = './pretrained/pvt_v2_b4.pth'
            if os.path.isfile(pretrained_path):
                save_model = torch.load(
                    pretrained_path,
                    map_location='cpu',
                )
                model_dict = self.bkbone.state_dict()
                state_dict = {
                    k: v
                    for k, v in save_model.items()
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                model_dict.update(state_dict)
                self.bkbone.load_state_dict(model_dict)
            else:
                print(
                    'Warning: PVTv2 pretrained weights not found at {}'.format(
                        pretrained_path
                    )
                )

        self.use_dinov2 = use_dinov2
        self.tsf = TSF(
            rgb_channels=enc_channels,
            use_dinov2=use_dinov2,
            dinov2_model=dinov2_model,
            dinov2_path=dinov2_path,
            dinov2_input_size=dinov2_input_size,
            freeze_dinov2=freeze_dinov2,
        )

        self.reduce_1 = Reduction(enc_channels[0], channel, RFB=False)
        self.reduce_2 = Reduction(enc_channels[1], channel, RFB=False)
        self.reduce_3 = Reduction(enc_channels[2], channel, RFB=False)
        self.reduce_4 = Reduction(enc_channels[3], channel, RFB=False)

        self.cca1 = CCA(channel)
        self.cca2 = CCA(channel)
        self.cca3 = CCA(channel)

        self.svr = SVR(channel)

        self.pre_fg0 = nn.Conv2d(channel, 1, 1)
        self.pre_fg1 = nn.Conv2d(channel, 1, 1)
        self.pre_fg2 = nn.Conv2d(channel, 1, 1)
        self.pre_fg3 = nn.Conv2d(channel, 1, 1)
        self.pre_fg4 = nn.Conv2d(channel, 1, 1)

    def forward(self, x, depth):
        output_size = x.size()[2:]

        rgb_features = self.bkbone(x)
        x1, x2, x3, x4 = self.tsf(rgb_features, x, depth)

        level_1 = self.reduce_1(x1)
        level_2 = self.reduce_2(x2)
        level_3 = self.reduce_3(x3)
        level_4 = self.reduce_4(x4)

        resized_4 = F.interpolate(level_4, size=level_3.size()[2:], mode='bilinear')
        merged_3 = self.cca1(level_3, resized_4)

        resized_3 = F.interpolate(merged_3, size=level_2.size()[2:], mode='bilinear')
        merged_2 = self.cca2(level_2, resized_3)

        resized_2 = F.interpolate(merged_2, size=level_1.size()[2:], mode='bilinear')
        merged_1 = self.cca3(level_1, resized_2)
        refined = self.svr(merged_1, level_1)

        return tuple(
            F.interpolate(head(feature), size=output_size, mode='bilinear')
            for head, feature in (
                (self.pre_fg0, resized_4),
                (self.pre_fg1, resized_3),
                (self.pre_fg2, resized_2),
                (self.pre_fg3, merged_1),
                (self.pre_fg4, refined),
            )
        )

if __name__ == '__main__':
    from time import time

    net = Network(imagenet_pretrained=False, use_dinov2=False)
    net.eval()

    dump_x = torch.randn(1, 3, 384, 384)
    dump_depth = torch.rand(1, 1, 384, 384)
    start = time()
    predictions = net(dump_x, dump_depth)
    print('forward time:', time() - start)
    print('prediction shapes:', [prediction.shape for prediction in predictions])
