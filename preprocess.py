import numpy as np
from PIL import Image


def invert(image):
    arr = np.asarray(image.convert("RGB"))
    return Image.fromarray((255 - arr).astype(np.uint8))


def densest_patch(image, patch_size=256):
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    if h < patch_size or w < patch_size:
        scale = patch_size / min(h, w)
        nh = max(patch_size, int(round(h * scale)))
        nw = max(patch_size, int(round(w * scale)))
        gray = np.asarray(image.convert("L").resize((nw, nh)))
        h, w = gray.shape
    ink = (gray > 128).astype(np.int64)
    best_x = 0
    best_y = 0
    best = -1
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            score = int(ink[y:y + patch_size, x:x + patch_size].sum())
            if score > best:
                best = score
                best_x = x
                best_y = y
    if best < 0:
        return image.resize((patch_size, patch_size))
    box = (best_x, best_y, best_x + patch_size, best_y + patch_size)
    return image.crop(box)


def preprocess(image, patch_size=256):
    inv = invert(image)
    patch = densest_patch(inv, patch_size)
    return inv, patch


def preprocess_path(path, patch_size=256):
    image = Image.open(path)
    return preprocess(image, patch_size)
