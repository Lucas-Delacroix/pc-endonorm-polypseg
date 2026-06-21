import csv
from pathlib import Path
from experiments.recall_safe_calibration.metrics_calibration import compute_metrics
from experiments.recall_safe_calibration.utils import ensure_new_file
DICE_TOLERANCE = 0.01
PRECISION_TOLERANCE = 0.05

def sweep_thresholds(logits, targets, *, thresholds, variants, ece_bins, lesion_dilation):
    rows = []
    for variant, temperature in variants.items():
        for threshold in thresholds:
            metrics = compute_metrics(logits, targets, threshold=threshold, temperature=temperature, ece_bins=ece_bins, lesion_dilation=lesion_dilation, include_boundary=False)
            rows.append({'variant': variant, **metrics})
    return rows

def select_recall_safe_threshold(rows, variant):
    variant_rows = [row for row in rows if row['variant'] == variant]
    baseline = min(variant_rows, key=lambda row: abs(float(row['threshold']) - 0.5))
    dice_floor = float(baseline['dice']) - DICE_TOLERANCE
    precision_floor = float(baseline['precision']) - PRECISION_TOLERANCE

    def best(candidates):
        if not candidates:
            return None
        return max(candidates, key=lambda row: (float(row['recall']), float(row['dice']), float(row['precision']), -abs(float(row['threshold']) - 0.5)))
    strict_candidates = [row for row in variant_rows if float(row['dice']) >= dice_floor and float(row['precision']) >= precision_floor]
    selected = best(strict_candidates)
    selected_by = 'dice_and_precision'
    if selected is None:
        dice_candidates = [row for row in variant_rows if float(row['dice']) >= dice_floor]
        selected = best(dice_candidates)
        selected_by = 'dice_only'
    if selected is None:
        selected = baseline
        selected_by = 'fallback_0.5'
    return {'variant': variant, 'threshold': float(selected['threshold']), 'selected_by': selected_by, 'rule': {'objective': 'maximize validation recall', 'dice_floor': dice_floor, 'precision_floor': precision_floor, 'dice_tolerance': DICE_TOLERANCE, 'precision_tolerance': PRECISION_TOLERANCE, 'fallback': 'threshold 0.5 if no threshold satisfies the Dice constraint'}, 'reference_threshold_0_5': baseline, 'selected_metrics': selected}

def write_sweep_csv(path, rows, *, overwrite=False):
    ensure_new_file(path, overwrite=overwrite)
    fieldnames = ['variant', 'threshold', 'temperature', 'dice', 'iou', 'recall', 'precision', 'f1', 'brier_score', 'ece_global', 'ece_foreground', 'specificity', 'tp', 'fp', 'fn', 'tn', 'n_images', 'n_pixels']
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

def build_selected_payload(rows, *, thresholds, variants, cache_metadata):
    selections = {variant: select_recall_safe_threshold(rows, variant) for variant in variants}
    return {'threshold_grid': thresholds, 'variants': variants, 'validation_cache_metadata': cache_metadata, 'selection_rule': {'primary': 'max Recall_val', 'constraint_1': 'Dice_val >= Dice_val@0.5 - 0.01', 'constraint_2': 'Precision_val >= Precision_val@0.5 - 0.05, if possible', 'fallback': 'use only Dice constraint; otherwise keep threshold 0.5'}, **selections}
