import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

ENCODER = 'vitl'
MODEL_CONFIG = {
    'vits': dict(encoder='vits', features=64, out_channels=[48, 96, 192, 384]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}[ENCODER]
MODEL_PATH = f'depth_anything_v2_{ENCODER}.pth'
INPUT_SIZE = 518
def infer_depth(model, raw_bgr, device):
    image, (h, w) = model.image2tensor(raw_bgr, INPUT_SIZE)
    image = image.to(device)
    with torch.no_grad():
        depth = model.forward(image)
        if isinstance(depth, tuple):
            depth = depth[0]
        elif isinstance(depth, dict):
            depth = list(depth.values())[-1]
        elif isinstance(depth, list):
            depth = depth[-1]

    if depth.dim() == 3:
        depth = depth[:, None]
    elif depth.dim() == 2:
        depth = depth[None, None]
    depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=True)
    return depth[0, 0].cpu().numpy()


def main():
    from utils.config import Config
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = Config()

    train_root = cfg.dp.train_root
    test_root = cfg.dp.test_root
    jobs = [
        (cfg.dp.train_imgs,
         os.path.join(train_root, 'Depth')),
        (cfg.dp.test_CHAMELEON_imgs,
         os.path.join(test_root, 'CHAMELEON', 'Depth')),
        (cfg.dp.test_CAMO_imgs,
         os.path.join(test_root, 'CAMO', 'Depth')),
        (cfg.dp.test_COD10K_imgs,
         os.path.join(test_root, 'COD10K', 'Depth')),
        (cfg.dp.test_NC4K_imgs,
         os.path.join(test_root, 'NC4K', 'Depth')),
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}（CPU 会很慢，建议开卡跑）')

    model = DepthAnythingV2(**MODEL_CONFIG)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(device).eval()

    for img_dir, depth_dir in jobs:
        if not os.path.isdir(img_dir):
            print(f'[跳过] 未找到: {img_dir}')
            continue
        os.makedirs(depth_dir, exist_ok=True)

        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        print(f'\n[处理] {img_dir} ({len(files)} 张)')
        done = 0
        for fname in tqdm(files):
            stem = os.path.splitext(fname)[0]
            depth_path = os.path.join(depth_dir, stem + '.png')
            if os.path.exists(depth_path):
                continue
            img = cv2.imread(os.path.join(img_dir, fname))
            if img is None:
                print(f'[警告] 读取失败: {fname}')
                continue
            depth = infer_depth(model, img, device)
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            cv2.imwrite(depth_path, (depth_norm * 65535).astype(np.uint16))
            done += 1
        print(f'[完成] 新增 {done} 张')

    print('\n[核对]（应为 train 4040；CHAMELEON 76；CAMO 250；COD10K 2026；NC4K 4121）')
    for img_dir, depth_dir in jobs:
        if not os.path.isdir(img_dir):
            continue
        n_img = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
        n_d = len(os.listdir(depth_dir)) if os.path.isdir(depth_dir) else 0
        print(f'{img_dir}: imgs {n_img} | Depth {n_d}')


if __name__ == '__main__':
    main()
