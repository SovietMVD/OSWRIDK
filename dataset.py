import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from preprocess import preprocess_path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _is_image(name):
    return name.lower().endswith(IMAGE_EXTS)


def collect_samples(root):
    samples = []
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        for name in sorted(os.listdir(cls_dir)):
            if _is_image(name):
                samples.append((os.path.join(cls_dir, name), label))
    return samples


def collect_ood(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if _is_image(name):
                paths.append(os.path.join(dirpath, name))
    return paths


def build_transform(size, mean=config.MEAN, std=config.STD):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class PairedDataset(Dataset):
    def __init__(self, samples, size):
        self.samples = samples
        self.transform = build_transform(size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        full, patch = preprocess_path(path, config.PATCH_SIZE)
        return self.transform(full), self.transform(patch), label
