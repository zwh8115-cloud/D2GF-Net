import os
import random

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def depth_loader(path):
    with open(path, "rb") as f:
        depth = Image.open(f)
        if depth.mode != "L":
            arr = np.array(depth).astype(np.float32)
            if arr.size and arr.max() > 255.0:
                arr = arr / 257.0
            depth = Image.fromarray(
                arr.clip(0, 255).astype(np.uint8),
                mode="L",
            )
        return depth


def cv_random_flip(img, label, grad, depth):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        grad = grad.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img, label, grad, depth


def randomCrop(image, label, grad, depth):
    border = min(30, max(1, min(image.size) // 4))
    image_width, image_height = image.size
    crop_win_width = np.random.randint(
        max(1, image_width - border),
        image_width + 1,
    )
    crop_win_height = np.random.randint(
        max(1, image_height - border),
        image_height + 1,
    )
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return (
        image.crop(random_region),
        label.crop(random_region),
        grad.crop(random_region),
        depth.crop(random_region),
    )


def randomRotation(image, label, grad, depth):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(
            random_angle,
            resample=Image.Resampling.BICUBIC,
        )
        label = label.rotate(
            random_angle,
            resample=Image.Resampling.NEAREST,
        )
        grad = grad.rotate(
            random_angle,
            resample=Image.Resampling.NEAREST,
        )
        depth = depth.rotate(
            random_angle,
            resample=Image.Resampling.BILINEAR,
        )
    return image, label, grad, depth


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    return ImageEnhance.Sharpness(image).enhance(sharp_intensity)


def randomPeper(img):
    img = np.array(img)
    noise_num = int(0.0015 * img.shape[0] * img.shape[1])
    for _ in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        img[rand_x, rand_y] = random.randint(0, 1) * 255
    return Image.fromarray(img)


def _normalize_stem(stem):
    for prefix in ("CAMO_", "COD10K_", "CHAMELEON_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _file_map(root):
    if not os.path.isdir(root):
        raise FileNotFoundError(
            "Missing dataset directory: {}".format(root)
        )
    result = {}
    for filename in os.listdir(root):
        stem, extension = os.path.splitext(filename)
        if extension.lower() in IMAGE_EXTENSIONS:
            result[_normalize_stem(stem)] = os.path.join(root, filename)
    return result


def _common_samples(image_root, gt_root, other_roots):
    maps = [_file_map(image_root), _file_map(gt_root)]
    maps.extend(_file_map(root) for root in other_roots)
    common_stems = set(maps[0])
    for file_map in maps[1:]:
        common_stems &= set(file_map)
    if not common_stems:
        raise RuntimeError(
            "No matching RGB/GT/Edge/Depth files were found"
        )

    all_stems = set().union(*(set(file_map) for file_map in maps))
    missing_count = len(all_stems - common_stems)
    if missing_count:
        print(
            ">>> warning: ignoring {} unmatched dataset files".format(
                missing_count
            )
        )

    return [
        tuple(file_map[stem] for file_map in maps)
        for stem in sorted(common_stems)
    ]


class CamObjDataset(data.Dataset):
    def __init__(
        self,
        image_root,
        gt_root,
        grad_root,
        depth_root,
        trainsize,
    ):
        self.trainsize = trainsize
        self.samples = _common_samples(
            image_root,
            gt_root,
            (grad_root, depth_root),
        )
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )
        self.depth_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
        self.size = len(self.samples)
        print(">>> training with {} samples".format(self.size))

    def __getitem__(self, index):
        image_path, gt_path, grad_path, depth_path = self.samples[index]
        image = self.rgb_loader(image_path)
        gt = self.binary_loader(gt_path)
        grad = self.binary_loader(grad_path)
        depth = depth_loader(depth_path)

        image, gt, grad, depth = cv_random_flip(
            image,
            gt,
            grad,
            depth,
        )
        image, gt, grad, depth = randomCrop(
            image,
            gt,
            grad,
            depth,
        )
        image, gt, grad, depth = randomRotation(
            image,
            gt,
            grad,
            depth,
        )

        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.mask_transform(gt)
        grad = self.mask_transform(grad)
        depth = self.depth_transform(depth)
        return image, gt, grad, depth

    @staticmethod
    def rgb_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    @staticmethod
    def binary_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("L")

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    gt_root,
    grad_root,
    depth_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
    worker_init_fn=None,
    generator=None,
):
    dataset = CamObjDataset(
        image_root,
        gt_root,
        grad_root,
        depth_root,
        trainsize,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.samples = _common_samples(
            image_root,
            gt_root,
            (depth_root,),
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.testsize, self.testsize),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.depth_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.testsize, self.testsize),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
        self.size = len(self.samples)
        self.index = 0

    def load_data(self):
        image_path, gt_path, depth_path = self.samples[self.index]
        image_pil = self.rgb_loader(image_path)
        image = self.transform(image_pil).unsqueeze(0)
        depth = self.depth_transform(
            depth_loader(depth_path)
        ).unsqueeze(0)
        gt = self.binary_loader(gt_path)
        name = os.path.splitext(os.path.basename(image_path))[0] + ".png"

        image_for_post = image_pil.resize(
            gt.size,
            resample=Image.Resampling.BILINEAR,
        )
        self.index = (self.index + 1) % self.size
        return image, depth, gt, name, np.array(image_for_post)

    @staticmethod
    def rgb_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    @staticmethod
    def binary_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("L")

    def __len__(self):
        return self.size
