import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.models.detectors.Net import Network
from utils.sdy_data_val import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument(
    '--testsize',
    type=int,
    default=448,
    help='testing size',
)
parser.add_argument(
    '--pth_path',
    type=str,
    default='./checkpoints/D2GF-Net/Net_epoch_best.pth',
)
parser.add_argument(
    '--rgb_dataset_root',
    type=str,
    default='./data/TestDataset',
    help='root containing RGB test datasets',
)
parser.add_argument(
    '--depth_dataset_root',
    type=str,
    default='./data/TestDataset/Depth',
    help='root containing generated depth test datasets',
)
parser.add_argument(
    '--datasets',
    type=str,
    default='CAMO,CHAMELEON,COD10K,NC4K',
    help='comma-separated dataset names',
)
parser.add_argument(
    '--gpu_id',
    type=str,
    default='0',
)
parser.add_argument(
    '--dinov2_path',
    type=str,
    default='./pretrained/dinov2_large.safetensors',
    help='path to the pretrained DINOv2 checkpoint (timm format)',
)
parser.add_argument(
    '--dinov2_model',
    type=str,
    default='vit_large_patch14_dinov2',
    help='DINOv2 timm model name',
)
parser.add_argument(
    '--dinov2_input_size',
    type=int,
    default=448,
    help='input size used by the DINOv2 branch',
)
parser.add_argument(
    '--disable_dinov2',
    action='store_true',
    help='disable the DINOv2 concat fusion branch',
)
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(opt.pth_path, map_location='cpu')
model = Network(
    use_dinov2=not opt.disable_dinov2,
    dinov2_model=opt.dinov2_model,
    dinov2_path=opt.dinov2_path,
    dinov2_input_size=opt.dinov2_input_size,
)
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint,
    strict=False,
)
if missing_keys or unexpected_keys:
    print(
        'checkpoint missing keys: {}, unexpected keys: {}'.format(
            len(missing_keys),
            len(unexpected_keys),
        )
    )
model.to(device)
model.eval()

checkpoint_dir = os.path.dirname(os.path.normpath(opt.pth_path))
save_path = os.path.join(
    './res',
    os.path.basename(checkpoint_dir),
    'final',
)

for data_name in [
    name.strip()
    for name in opt.datasets.split(',')
    if name.strip()
]:
    data_path = os.path.join(
        opt.rgb_dataset_root,
        data_name,
    )
    nested_test_path = os.path.join(data_path, 'test')
    if os.path.isdir(nested_test_path):
        data_path = nested_test_path
    depth_data_path = os.path.join(
        opt.depth_dataset_root,
        data_name,
        'test',
    )
    image_root = os.path.join(data_path, 'Imgs')
    gt_root = os.path.join(data_path, 'GT')
    depth_root = os.path.join(data_path, 'Depth')
    if not os.path.isdir(depth_root):
        depth_root = os.path.join(depth_data_path, 'Depth')
    save_path_final = os.path.join(save_path, data_name)
    os.makedirs(save_path_final, exist_ok=True)

    test_loader = test_dataset(
        image_root,
        gt_root,
        depth_root,
        opt.testsize,
    )

    with torch.no_grad():
        for _ in range(test_loader.size):
            image, depth, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8
            image = image.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            preds = model(image, depth)
            res = F.interpolate(
                preds[-2] + preds[-1],
                size=gt.shape,
                mode='bilinear',
                align_corners=False,
            )
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (
                res.max() - res.min() + 1e-8
            )
            res = (255 * res).astype(np.uint8)
            cv2.imwrite(
                os.path.join(save_path_final, name),
                res,
            )
            print('{} - {}'.format(data_name, name))

    print('{} Finish!'.format(data_name))
