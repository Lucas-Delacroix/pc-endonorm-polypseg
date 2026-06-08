import numpy as np
from skimage.morphology import black_tophat, disk, white_tophat

from preprocessing.common import lab_luminance, normalize

MORPH_RADII = (3, 7, 15)


def compute_morph_map(rgb, config=None):
    config = config or {}
    radii = config.get("radii", MORPH_RADII)

    luminance = lab_luminance(rgb)
    top_hat = np.mean([white_tophat(luminance, disk(r)) for r in radii], axis=0)
    bottom_hat = np.mean([black_tophat(luminance, disk(r)) for r in radii], axis=0)
    return normalize(top_hat + bottom_hat)
