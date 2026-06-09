import cv2
import numpy as np

CLIP_LIMIT = 2.0
TILE_GRID_SIZE = 8


def compute_clahe_rgb(rgb, config=None):
    config = config or {}
    clip_limit = config.get("clip_limit", CLIP_LIMIT)
    tile_size = config.get("tile_grid_size", TILE_GRID_SIZE)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced.astype(np.float32) / 255.0
