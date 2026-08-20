import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


def binarize(image):
    gray = np.asarray(image.convert("L"))
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    weight_b = 0.0
    best_thresh = 0
    best_var = -1.0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t
    binary = (gray > best_thresh).astype(np.uint8) * 255
    return Image.fromarray(binary)


def cutmix(image1, image2, alpha=1.0):
    w, h = image1.size
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    a = np.asarray(image1.convert("RGB"))
    b = np.asarray(image2.resize((w, h)).convert("RGB"))
    a[bby1:bby2, bbx1:bbx2] = b[bby1:bby2, bbx1:bbx2]
    return Image.fromarray(a), lam


def _perlin_noise(size):
    w, h = size
    noise = (torch.rand((3, h, w)) * 255).byte()
    return F.to_pil_image(noise)


def pixmix(image, mix_image, alpha=0.5):
    ops = [
        lambda x: F.solarize(x, threshold=128),
        lambda x: F.adjust_sharpness(x, sharpness_factor=3.0),
        lambda x: F.autocontrast(x),
        lambda x: F.rotate(x, angle=45),
        lambda x: _perlin_noise(x.size),
    ]
    transformed = image.convert("RGB")
    for _ in range(np.random.randint(3, 6)):
        op = ops[np.random.randint(len(ops))]
        try:
            transformed = op(transformed.convert("RGB"))
        except Exception:
            transformed = _perlin_noise(transformed.size)
    mix = mix_image.resize(image.size).convert("RGB")
    mask = np.random.beta(alpha, alpha, size=(image.size[1], image.size[0]))
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    mixed = np.asarray(transformed.convert("RGB"), dtype=np.float32)
    blended = original * (1.0 - mask[..., None]) + mixed * mask[..., None]
    return Image.fromarray(blended.astype(np.uint8))
