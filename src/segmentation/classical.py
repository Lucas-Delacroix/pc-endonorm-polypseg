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

def _remove_specular(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    highlight = ((hsv[..., 2] >= SPECULAR_VALUE_MIN) & (hsv[..., 1] <= SPECULAR_SAT_MAX)).astype(np.uint8)
    if not highlight.any():
        return bgr
    kernel = np.ones((SPECULAR_DILATE, SPECULAR_DILATE), np.uint8)
    highlight = cv2.dilate(highlight, kernel)
    return cv2.inpaint(bgr, highlight, INPAINT_RADIUS, cv2.INPAINT_TELEA)

def _valid_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return (gray > BORDER_DARK_MAX).astype(np.uint8)

def _redness_channel(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    channel = lab[..., A_CHANNEL]
    channel = cv2.createCLAHE(CLAHE_CLIP, (CLAHE_GRID, CLAHE_GRID)).apply(channel)
    return cv2.GaussianBlur(channel, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)

def _morphology(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERATIONS)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS)

def _fill_holes(binary):
    flood = binary.copy()
    height, width = binary.shape
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    return (binary | 1 - flood).astype(np.uint8)

def _largest_component(binary):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    if areas[largest - 1] < MIN_AREA_RATIO * binary.size:
        return np.zeros_like(binary)
    return (labels == largest).astype(np.uint8)

def _otsu(channel, valid):
    values = channel[valid > 0]
    if values.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((channel >= threshold) & (valid > 0)).astype(np.uint8)

def segment(bgr):
    original_height, original_width = bgr.shape[:2]
    working = cv2.resize(bgr, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    working = _remove_specular(working)
    valid = _valid_mask(working)
    channel = _redness_channel(working)
    binary = _otsu(channel, valid)
    binary = _morphology(binary)
    binary = _fill_holes(binary)
    binary = _largest_component(binary)
    resized = cv2.resize(binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return (resized * FOREGROUND).astype(np.uint8)
