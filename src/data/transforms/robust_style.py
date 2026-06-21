from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

JITTER = {"brightness": 0.25, "contrast": 0.25, "saturation": 0.25, "hue": 0.05, "p": 0.8}
GAMMA_LIMIT = (70, 140)
HSV_LIMITS = {"hue_shift_limit": 10, "sat_shift_limit": 25, "val_shift_limit": 20}
BLUR_LIMIT = (3, 5)
SHIFT = (-0.1, 0.1)
SCALE = (0.8, 1.2)
ROTATE = (-10, 10)
PERSPECTIVE_SCALE = (0.05, 0.1)
FOURIER = {"p": 0.5, "beta": 0.05, "strength": 0.15, "low_freq_only": True}


class FourierAmplitudeRandomization(A.ImageOnlyTransform):
    def __init__(self, beta=0.05, strength=0.15, low_freq_only=True, p=0.5):
        super().__init__(p=p)
        self.beta = beta
        self.strength = strength
        self.low_freq_only = low_freq_only

    def _lowpass_weight(self, height: int, width: int) -> np.ndarray:
        if not self.low_freq_only:
            return np.ones((height, width), dtype=np.float32)
        rows = np.arange(height).reshape(-1, 1)
        cols = np.arange(width).reshape(1, -1)
        center_row = (height - 1) / 2.0
        center_col = (width - 1) / 2.0
        squared_distance = (rows - center_row) ** 2 + (cols - center_col) ** 2
        sigma = max(self.beta * min(height, width), 1.0)
        return np.exp(-squared_distance / (2.0 * sigma**2)).astype(np.float32)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.astype(np.float32) / 255.0
        height, width = image.shape[:2]
        weight = self._lowpass_weight(height, width)
        output = np.empty_like(image)

        for channel in range(image.shape[2]):
            spectrum = np.fft.fftshift(np.fft.fft2(image[..., channel]))
            amplitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            factor = np.random.uniform(1.0 - self.strength, 1.0 + self.strength)
            scale_map = 1.0 + (factor - 1.0) * weight
            reconstructed = np.fft.ifft2(np.fft.ifftshift(amplitude * scale_map * np.exp(1j * phase))).real
            output[..., channel] = reconstructed

        output = np.clip(output, 0.0, 1.0)
        return (output * 255.0).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("beta", "strength", "low_freq_only")


def is_on(config: dict, key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, dict):
        return bool(value.get("enabled", default))
    return bool(value)


def strong_style_transforms() -> list:
    return [
        A.ColorJitter(**JITTER),
        A.RandomGamma(gamma_limit=GAMMA_LIMIT, p=0.5),
        A.HueSaturationValue(**HSV_LIMITS, p=0.5),
        A.GaussianBlur(blur_limit=BLUR_LIMIT, p=0.2),
        A.GaussNoise(p=0.2),
    ]


def fourier_transform() -> FourierAmplitudeRandomization:
    return FourierAmplitudeRandomization(**FOURIER)


def build_geometric_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=SHIFT, scale=SCALE, rotate=ROTATE, p=0.5),
        A.Perspective(scale=PERSPECTIVE_SCALE, p=0.5),
    ])


def build_appearance_transforms(augmentation: dict | None) -> A.Compose:
    augmentation = augmentation or {}
    transforms = []
    if is_on(augmentation, "strong_style", default=True):
        transforms += strong_style_transforms()
    if is_on(augmentation, "fourier"):
        transforms.append(fourier_transform())
    return A.Compose(transforms)


def build_normalize_transforms() -> A.Compose:
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def build_robust_train_transforms(image_size: int, augmentation: dict | None) -> A.Compose:
    return A.Compose([
        *build_geometric_transforms(image_size).transforms,
        *build_appearance_transforms(augmentation).transforms,
        *build_normalize_transforms().transforms,
    ])
