from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FOURIER_DEFAULTS = {"p": 0.5, "beta": 0.05, "strength": 0.15, "low_freq_only": True}
RANDCONV_DEFAULTS = {
    "p": 0.3,
    "kernel_sizes": (1, 3, 5),
    "mix_alpha_min": 0.1,
    "mix_alpha_max": 0.35,
    "depthwise": True,
    "normalize_kernel": True,
    "normalize_output": True,
    "same_kernel_per_channel": False,
}
KERNEL_NORM_EPS = 1e-8


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
            perturbed = amplitude * scale_map

            reconstructed = np.fft.ifft2(np.fft.ifftshift(perturbed * np.exp(1j * phase))).real
            output[..., channel] = reconstructed

        output = np.clip(output, 0.0, 1.0)
        return (output * 255.0).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("beta", "strength", "low_freq_only")


class RandConvTransform(A.ImageOnlyTransform):
    def __init__(
        self,
        kernel_sizes=(1, 3, 5),
        mix_alpha_min=0.1,
        mix_alpha_max=0.35,
        depthwise=True,
        normalize_kernel=True,
        normalize_output=True,
        same_kernel_per_channel=False,
        p=0.3,
    ):
        super().__init__(p=p)
        self.kernel_sizes = tuple(kernel_sizes)
        self.mix_alpha_min = mix_alpha_min
        self.mix_alpha_max = mix_alpha_max
        self.depthwise = depthwise
        self.normalize_kernel = normalize_kernel
        self.normalize_output = normalize_output
        self.same_kernel_per_channel = same_kernel_per_channel

    def _random_kernel(self, kernel_size: int) -> np.ndarray:
        kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
        if self.normalize_kernel:
            kernel = kernel / (np.abs(kernel).sum() + KERNEL_NORM_EPS)
        return kernel

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.astype(np.float32) / 255.0
        kernel_size = int(np.random.choice(self.kernel_sizes))

        if self.same_kernel_per_channel or not self.depthwise:
            kernel = self._random_kernel(kernel_size)
            convolved = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
        else:
            convolved = np.empty_like(image)
            for channel in range(image.shape[2]):
                kernel = self._random_kernel(kernel_size)
                convolved[..., channel] = cv2.filter2D(
                    image[..., channel], -1, kernel, borderType=cv2.BORDER_REFLECT
                )

        alpha = np.random.uniform(self.mix_alpha_min, self.mix_alpha_max)
        output = (1.0 - alpha) * image + alpha * convolved
        if self.normalize_output:
            output = np.clip(output, 0.0, 1.0)
        return (output * 255.0).astype(np.uint8)

    def get_transform_init_args_names(self):
        return (
            "kernel_sizes",
            "mix_alpha_min",
            "mix_alpha_max",
            "depthwise",
            "normalize_kernel",
            "normalize_output",
            "same_kernel_per_channel",
        )


def build_randconv_transform(config: dict) -> RandConvTransform:
    return RandConvTransform(
        kernel_sizes=config.get("kernel_sizes", RANDCONV_DEFAULTS["kernel_sizes"]),
        mix_alpha_min=config.get("mix_alpha_min", RANDCONV_DEFAULTS["mix_alpha_min"]),
        mix_alpha_max=config.get("mix_alpha_max", RANDCONV_DEFAULTS["mix_alpha_max"]),
        depthwise=config.get("depthwise", RANDCONV_DEFAULTS["depthwise"]),
        normalize_kernel=config.get("normalize_kernel", RANDCONV_DEFAULTS["normalize_kernel"]),
        normalize_output=config.get("normalize_output", RANDCONV_DEFAULTS["normalize_output"]),
        same_kernel_per_channel=config.get("same_kernel_per_channel", RANDCONV_DEFAULTS["same_kernel_per_channel"]),
        p=config.get("p", RANDCONV_DEFAULTS["p"]),
    )


def _strong_color_transforms(config: dict) -> list:
    transforms = []

    jitter = config.get("color_jitter", {})
    if jitter.get("enabled", True):
        transforms.append(A.ColorJitter(
            brightness=jitter.get("brightness", 0.25),
            contrast=jitter.get("contrast", 0.25),
            saturation=jitter.get("saturation", 0.25),
            hue=jitter.get("hue", 0.05),
            p=jitter.get("p", 0.8),
        ))

    gamma = config.get("random_gamma", {})
    if gamma.get("enabled", True):
        transforms.append(A.RandomGamma(
            gamma_limit=tuple(gamma.get("gamma_limit", (70, 140))),
            p=gamma.get("p", 0.5),
        ))

    hsv = config.get("hue_saturation_value", {})
    if hsv.get("enabled", True):
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=hsv.get("hue_shift_limit", 10),
            sat_shift_limit=hsv.get("sat_shift_limit", 25),
            val_shift_limit=hsv.get("val_shift_limit", 20),
            p=hsv.get("p", 0.5),
        ))

    blur = config.get("gaussian_blur", {})
    if blur.get("enabled", True):
        transforms.append(A.GaussianBlur(
            blur_limit=tuple(blur.get("blur_limit", (3, 5))),
            p=blur.get("p", 0.2),
        ))

    noise = config.get("gaussian_noise", {})
    if noise.get("enabled", True):
        transforms.append(A.GaussNoise(p=noise.get("p", 0.2)))

    motion = config.get("motion_blur", {})
    if motion.get("enabled", False):
        transforms.append(A.MotionBlur(blur_limit=motion.get("blur_limit", 3), p=motion.get("p", 0.1)))

    grayscale = config.get("grayscale", {})
    if grayscale.get("enabled", False):
        transforms.append(A.ToGray(p=grayscale.get("p", 0.05)))

    return transforms


def build_robust_train_transforms(image_size: int, augmentation: dict | None) -> A.Compose:
    augmentation = augmentation or {}
    strong = augmentation.get("strong_style", {})
    fourier = augmentation.get("fourier", {})
    randconv = augmentation.get("randconv", {})

    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
    ]

    if strong.get("enabled", True):
        transforms += _strong_color_transforms(strong)

    if fourier.get("enabled", True):
        transforms.append(FourierAmplitudeRandomization(
            beta=fourier.get("beta", FOURIER_DEFAULTS["beta"]),
            strength=fourier.get("strength", FOURIER_DEFAULTS["strength"]),
            low_freq_only=fourier.get("low_freq_only", FOURIER_DEFAULTS["low_freq_only"]),
            p=fourier.get("p", FOURIER_DEFAULTS["p"]),
        ))

    if randconv.get("enabled", False):
        transforms.append(build_randconv_transform(randconv))

    transforms += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(transforms)


def build_geometric_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
    ])


def build_appearance_transforms(augmentation: dict | None) -> A.Compose:
    augmentation = augmentation or {}
    strong = augmentation.get("strong_style", {})
    fourier = augmentation.get("fourier", {})
    randconv = augmentation.get("randconv", {})

    transforms = []
    if strong.get("enabled", True):
        transforms += _strong_color_transforms(strong)
    if fourier.get("enabled", True):
        transforms.append(FourierAmplitudeRandomization(
            beta=fourier.get("beta", FOURIER_DEFAULTS["beta"]),
            strength=fourier.get("strength", FOURIER_DEFAULTS["strength"]),
            low_freq_only=fourier.get("low_freq_only", FOURIER_DEFAULTS["low_freq_only"]),
            p=fourier.get("p", FOURIER_DEFAULTS["p"]),
        ))
    if randconv.get("enabled", False):
        transforms.append(build_randconv_transform(randconv))

    return A.Compose(transforms)


def build_normalize_transforms() -> A.Compose:
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
