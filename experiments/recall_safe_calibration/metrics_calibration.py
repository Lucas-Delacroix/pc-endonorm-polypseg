import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
HD95_PERCENTILE = 95
BOUNDARY_F1_TOLERANCE = 0.0075
SMOOTH = 1e-06

def sigmoid_temperature(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError(f'Temperature must be positive, got {temperature}.')
    scaled = np.clip(logits.astype(np.float64) / float(temperature), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)

def confusion_counts(prediction, target):
    pred = prediction.astype(bool).ravel()
    gt = target.astype(bool).ravel()
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.float64)
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum(dtype=np.float64)
    return {'tp': float(tp), 'fp': float(fp), 'fn': float(fn), 'tn': float(tn)}

def metrics_from_counts(counts):
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    tn = counts['tn']
    precision = (tp + SMOOTH) / (tp + fp + SMOOTH)
    recall = (tp + SMOOTH) / (tp + fn + SMOOTH)
    f1 = 2.0 * precision * recall / (precision + recall + SMOOTH)
    return {'dice': float((2.0 * tp + SMOOTH) / (2.0 * tp + fp + fn + SMOOTH)), 'iou': float((tp + SMOOTH) / (tp + fp + fn + SMOOTH)), 'recall': float(recall), 'precision': float(precision), 'f1': float(f1), 'specificity': float((tn + SMOOTH) / (tn + fp + SMOOTH))}

def calibration_bins(probabilities, labels, n_bins):
    probs = probabilities.astype(np.float64).ravel()
    y = labels.astype(np.float64).ravel()
    if probs.size != y.size:
        raise ValueError('probabilities and labels must have the same number of pixels.')
    if probs.size == 0:
        return (float('nan'), [])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(probs.size)
    ece = 0.0
    bins = []
    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append({'bin_lower': lower, 'bin_upper': upper, 'count': 0, 'confidence': float('nan'), 'empirical_positive_rate': float('nan'), 'gap': float('nan')})
            continue
        confidence = float(probs[mask].mean())
        empirical_rate = float(y[mask].mean())
        gap = abs(confidence - empirical_rate)
        ece += count / total * gap
        bins.append({'bin_lower': lower, 'bin_upper': upper, 'count': count, 'confidence': confidence, 'empirical_positive_rate': empirical_rate, 'gap': gap})
    return (float(ece), bins)

def foreground_region(targets, dilation_iterations=0):
    foreground = targets.astype(bool)
    if dilation_iterations <= 0:
        return foreground
    dilated = np.zeros_like(foreground, dtype=bool)
    for index in range(foreground.shape[0]):
        dilated[index] = binary_dilation(foreground[index], iterations=dilation_iterations)
    return dilated

def surface_pixels(mask):
    return np.logical_xor(mask, binary_erosion(mask))

def boundary_metrics_single(prediction, target):
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    if not pred.any() and (not gt.any()):
        return {'hd95': 0.0, 'assd': 0.0, 'boundary_f1': 1.0}
    empty_penalty = float(np.hypot(*gt.shape))
    if not pred.any() or not gt.any():
        return {'hd95': empty_penalty, 'assd': empty_penalty, 'boundary_f1': 0.0}
    pred_surface = surface_pixels(pred)
    gt_surface = surface_pixels(gt)
    pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]
    hd95 = max(float(np.percentile(pred_to_gt, HD95_PERCENTILE)), float(np.percentile(gt_to_pred, HD95_PERCENTILE)))
    assd = float((pred_to_gt.sum() + gt_to_pred.sum()) / (pred_to_gt.size + gt_to_pred.size))
    tolerance = max(1.0, BOUNDARY_F1_TOLERANCE * empty_penalty)
    boundary_precision = float((pred_to_gt <= tolerance).mean())
    boundary_recall = float((gt_to_pred <= tolerance).mean())
    boundary_f1 = 2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if boundary_precision + boundary_recall > 0 else 0.0
    return {'hd95': hd95, 'assd': assd, 'boundary_f1': boundary_f1}

def mean_boundary_metrics(predictions, targets):
    totals = {'hd95': 0.0, 'assd': 0.0, 'boundary_f1': 0.0}
    for prediction, target in zip(predictions, targets, strict=True):
        values = boundary_metrics_single(prediction, target)
        for key in totals:
            totals[key] += values[key]
    count = max(1, int(predictions.shape[0]))
    return {key: float(value / count) for key, value in totals.items()}

def compute_metrics(logits, targets, *, threshold, temperature=1.0, ece_bins=15, lesion_dilation=0, include_boundary=True):
    probabilities = sigmoid_temperature(logits, temperature)
    target_bool = targets.astype(bool)
    predictions = probabilities >= threshold
    counts = confusion_counts(predictions, target_bool)
    metrics = metrics_from_counts(counts)
    labels = target_bool.astype(np.float32)
    ece_global, _ = calibration_bins(probabilities, labels, ece_bins)
    fg_mask = foreground_region(target_bool, lesion_dilation)
    if fg_mask.any():
        ece_foreground, _ = calibration_bins(probabilities[fg_mask], labels[fg_mask], ece_bins)
    else:
        ece_foreground = float('nan')
    metrics.update({'threshold': float(threshold), 'temperature': float(temperature), 'brier_score': float(np.mean((probabilities.astype(np.float64) - labels) ** 2)), 'ece_global': float(ece_global), 'ece_foreground': float(ece_foreground), 'n_images': int(targets.shape[0]), 'n_pixels': int(targets.size), **counts})
    if include_boundary:
        metrics.update(mean_boundary_metrics(predictions, target_bool))
    return metrics

def reliability_rows(logits, targets, *, temperature, ece_bins):
    probabilities = sigmoid_temperature(logits, temperature)
    _, bins = calibration_bins(probabilities, targets.astype(np.float32), ece_bins)
    return bins
