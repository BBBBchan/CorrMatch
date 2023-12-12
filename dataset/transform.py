import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torch
from torchvision import transforms


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize_certain(img, ratio_range):
    w, h = img.size
    ow = int(w * ratio_range + 0.5)
    oh = int(h * ratio_range + 0.5)
    img = img.resize((ow, oh), Image.BILINEAR)
    return img

def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)


def img_aug_identity(img, scale=None):
    return img


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.05, 0.95], p=0.2):
    if random.random() < p:
        min_v, max_v = min(scale), max(scale)
        v = float(max_v - min_v) * random.random()
        v = max_v - v
        return ImageEnhance.Contrast(img).enhance(v)
    else:
        return img


def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.solarize(img, v)


def get_augment_list():
    l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.05, 0.95]),
        (img_aug_brightness, [0.05, 0.95]),
        (img_aug_color, [0.05, 0.95]),
        (img_aug_sharpness, [0.05, 0.95]),
        (img_aug_posterize, [4, 8]),
        (img_aug_solarize, [1, 256]),
        (img_aug_hue, [0, 0.5])
    ]
    return l


class strong_img_aug:
    def __init__(self, num_augs=4, flag_using_random_num=True):
        self.n = num_augs
        self.augment_list = get_augment_list()
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num = self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img