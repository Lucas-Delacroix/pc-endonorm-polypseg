import cv2
import numpy as np
from skimage.morphology import black_tophat, dilation, disk, opening, white_tophat

from preprocessing.common import EPS, normalize

GAUSSIAN_SIGMA = 25
ILLUMINATION_BLEND = 0.5
SPECULAR_VALUE_MIN = 0.85
SPECULAR_SATURATION_MAX = 0.35
SPECULAR_TOPHAT_MIN = 0.4
TOPHAT_RADIUS = 7
OPENING_RADIUS = 1
DILATION_RADIUS = 2
INPAINT_RADIUS = 3


def compute_endonorm_rgb(rgb, config=None):
    config = config or {}
    sigma = config.get("sigma", GAUSSIAN_SIGMA)
    alpha = config.get("alpha", ILLUMINATION_BLEND)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[:, :, 0] / 255.0

    illumination = cv2.GaussianBlur(luminance, (0, 0), sigma)
    corrected = normalize(luminance / (illumination + EPS))
    blended = alpha * corrected + (1.0 - alpha) * luminance

    lab[:, :, 0] = np.clip(blended * 255.0, 0, 255)
    corrected_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    top_hat = normalize(white_tophat(luminance, disk(TOPHAT_RADIUS)))
    bottom_hat = normalize(black_tophat(luminance, disk(TOPHAT_RADIUS)))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    specular = (value > SPECULAR_VALUE_MIN) & (saturation < SPECULAR_SATURATION_MAX) & (top_hat > SPECULAR_TOPHAT_MIN)
    specular = dilation(opening(specular, disk(OPENING_RADIUS)), disk(DILATION_RADIUS))

    inpainted = cv2.inpaint(corrected_rgb, specular.astype(np.uint8) * 255, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    image_norm = inpainted.astype(np.float32) / 255.0

    debug = {
        "L": luminance,
        "illumination_field": normalize(illumination),
        "L_corrected": corrected,
        "top_hat": top_hat,
        "bottom_hat": bottom_hat,
        "specular_mask": specular.astype(np.float32),
    }
    return image_norm, debug
