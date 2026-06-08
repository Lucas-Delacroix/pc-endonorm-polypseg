import cv2
import numpy as np

EPS = 1e-6


def normalize(array):
    array = array.astype(np.float32)
    low, high = float(array.min()), float(array.max())
    if high - low < EPS:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def lab_luminance(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) / 255.0
