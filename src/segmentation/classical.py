from __future__ import annotations

import cv2
import numpy as np

WORK_SIZE = 352
SPECULAR_VALUE_MIN = 200
SPECULAR_SAT_MAX = 60
INPAINT_RADIUS = 5
SPECULAR_DILATE = 5
BORDER_DARK_MAX = 40
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
GAUSSIAN_KERNEL = 5
MORPH_KERNEL = 7
CLOSE_ITERATIONS = 2
OPEN_ITERATIONS = 1
MIN_AREA_RATIO = 0.01
A_CHANNEL = 1
FOREGROUND = 255
ADAPTIVE_BLOCK = 51
ADAPTIVE_C = 5
WATERSHED_FG_RATIO = 0.4
WATERSHED_BG_DILATE = 3

METHODS = ("otsu", "adaptive", "watershed")


def _remove_specular(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    highlight = ((hsv[..., 2] >= SPECULAR_VALUE_MIN) & (hsv[..., 1] <= SPECULAR_SAT_MAX)).astype(np.uint8)
    if not highlight.any():
        return bgr
    kernel = np.ones((SPECULAR_DILATE, SPECULAR_DILATE), np.uint8)
    highlight = cv2.dilate(highlight, kernel)
    return cv2.inpaint(bgr, highlight, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def _valid_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return (gray > BORDER_DARK_MAX).astype(np.uint8)


def _redness_channel(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    channel = lab[..., A_CHANNEL]
    channel = cv2.createCLAHE(CLAHE_CLIP, (CLAHE_GRID, CLAHE_GRID)).apply(channel)
    return cv2.GaussianBlur(channel, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)


def _morphology(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERATIONS)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    flood = binary.copy()
    height, width = binary.shape
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    return (binary | (1 - flood)).astype(np.uint8)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    if areas[largest - 1] < MIN_AREA_RATIO * binary.size:
        return np.zeros_like(binary)
    return (labels == largest).astype(np.uint8)


def _otsu(channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = channel[valid > 0]
    if values.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((channel >= threshold) & (valid > 0)).astype(np.uint8)


def _adaptive(channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        channel, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK, -ADAPTIVE_C
    )
    return (binary & (valid > 0)).astype(np.uint8)


def _watershed(bgr: np.ndarray, channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    seed = _morphology(_otsu(channel, valid))
    if not seed.any():
        return seed
    distance = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, WATERSHED_FG_RATIO * distance.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    sure_bg = cv2.dilate(seed, kernel, iterations=WATERSHED_BG_DILATE)
    unknown = cv2.subtract(sure_bg, sure_fg)
    count, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0
    markers = cv2.watershed(bgr, markers)
    return ((markers > 1) & (valid > 0)).astype(np.uint8)


def segment(bgr: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Options: {METHODS}")

    original_height, original_width = bgr.shape[:2]
    working = cv2.resize(bgr, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    working = _remove_specular(working)
    valid = _valid_mask(working)
    channel = _redness_channel(working)

    if method == "watershed":
        binary = _watershed(working, channel, valid)
    elif method == "adaptive":
        binary = _adaptive(channel, valid)
    else:
        binary = _otsu(channel, valid)

    binary = _morphology(binary)
    binary = _fill_holes(binary)
    binary = _largest_component(binary)

    resized = cv2.resize(binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return (resized * FOREGROUND).astype(np.uint8)
