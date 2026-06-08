import cv2
import numpy as np

from preprocessing.common import normalize

NSCALE = 4
NORIENT = 6
MIN_WAVELENGTH = 3.0
MULT = 2.1
SIGMA_ONF = 0.55
K_NOISE = 2.0
CUTOFF = 0.5
G_SHARP = 10.0
LOWPASS_CUTOFF = 0.45
LOWPASS_ORDER = 15
EPS = 1e-4
CLIP_PERCENTILES = (1.0, 99.0)


def _freq_range(n):
    if n % 2:
        return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1) / (n - 1)
    return np.arange(-n / 2, n / 2) / n


def _frequency_grid(rows, cols):
    return np.meshgrid(_freq_range(cols), _freq_range(rows))


def _lowpass(rows, cols):
    x, y = _frequency_grid(rows, cols)
    radius = np.sqrt(x**2 + y**2)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / LOWPASS_CUTOFF) ** (2 * LOWPASS_ORDER)))


def _phase_congruency(gray):
    rows, cols = gray.shape
    image_fft = np.fft.fft2(gray)

    x, y = _frequency_grid(rows, cols)
    radius = np.fft.ifftshift(np.sqrt(x**2 + y**2))
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    radius[0, 0] = 1.0
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)

    lowpass = _lowpass(rows, cols)
    log_gabor = []
    for scale in range(NSCALE):
        center = 1.0 / (MIN_WAVELENGTH * MULT**scale)
        gabor = np.exp(-(np.log(radius / center) ** 2) / (2 * np.log(SIGMA_ONF) ** 2)) * lowpass
        gabor[0, 0] = 0.0
        log_gabor.append(gabor)

    noise_gain = (1.0 - (1.0 / MULT) ** NSCALE) / (1.0 - 1.0 / MULT)
    congruency = np.zeros((rows, cols))

    for orient in range(NORIENT):
        angle = orient * np.pi / NORIENT
        ds = sin_theta * np.cos(angle) - cos_theta * np.sin(angle)
        dc = cos_theta * np.cos(angle) + sin_theta * np.sin(angle)
        dtheta = np.minimum(np.abs(np.arctan2(ds, dc)) * NORIENT / 2.0, np.pi)
        spread = (np.cos(dtheta) + 1.0) / 2.0

        responses = [np.fft.ifft2(image_fft * (gabor * spread)) for gabor in log_gabor]
        amplitude = [np.abs(r) for r in responses]
        sum_amp = np.sum(amplitude, axis=0)
        max_amp = np.max(amplitude, axis=0)
        sum_real = np.sum([r.real for r in responses], axis=0)
        sum_imag = np.sum([r.imag for r in responses], axis=0)

        magnitude = np.sqrt(sum_real**2 + sum_imag**2) + EPS
        mean_real, mean_imag = sum_real / magnitude, sum_imag / magnitude
        energy = sum(
            r.real * mean_real + r.imag * mean_imag - np.abs(r.real * mean_imag - r.imag * mean_real)
            for r in responses
        )

        tau = np.median(amplitude[0]) / np.sqrt(np.log(4.0))
        threshold = tau * noise_gain * (np.sqrt(np.pi / 2.0) + K_NOISE * np.sqrt((4.0 - np.pi) / 2.0))
        energy = np.maximum(energy - threshold, 0.0)

        width = (sum_amp / (max_amp + EPS) - 1.0) / (NSCALE - 1)
        weight = 1.0 / (1.0 + np.exp(G_SHARP * (CUTOFF - width)))
        congruency += weight * energy / (sum_amp + EPS)

    return congruency / NORIENT


def compute_phase_congruency_map(rgb, config=None):
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float64) / 255.0
    pc = _phase_congruency(luminance)
    low, high = np.percentile(pc, CLIP_PERCENTILES)
    return normalize(np.clip(pc, low, high)).astype(np.float32)
