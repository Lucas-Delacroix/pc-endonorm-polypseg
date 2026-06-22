import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
MODEL_NAMES = {'esfpnet': 'ESFPNet'}
HD95_PERCENTILE = 95
BOUNDARY_F1_TOLERANCE = 0.0075
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-root', default='outputs/predictions')
    parser.add_argument('--data-root', default='data/raw/kvasir-seg')
    parser.add_argument('--split-file', default='data/splits/kvasir_split.json')
    parser.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--models', nargs='*', default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', default='outputs/tables')
    parser.add_argument('--output-name', default='table2')
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--allow-missing', action='store_true')
    return parser.parse_args()

def load_split_indices(split_file, split):
    with open(split_file) as file:
        return json.load(file)[split]

def match_mask(masks_dir, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = masks_dir / f'{stem}{extension}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mask found for '{stem}' in {masks_dir}")

def load_samples(data_root, split_file, split, external):
    images_dir = data_root / 'images'
    masks_dir = data_root / 'masks'
    if external:
        images = sorted((p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
        return [(image, match_mask(masks_dir, image.stem)) for image in images]
    images = sorted(images_dir.glob('*.jpg'))
    return [(images[idx], masks_dir / images[idx].name) for idx in load_split_indices(split_file, split)]

def model_dirs(predictions_root, selected):
    if selected:
        return [predictions_root / name for name in selected]
    return sorted((path for path in predictions_root.iterdir() if path.is_dir()))

def build_prediction_index(prediction_dir):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return {path.stem: path for path in prediction_dir.rglob('*') if path.is_file() and path.suffix.lower() in image_extensions}

def read_binary_mask(path, threshold, target_shape=None):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Invalid mask: {path}')
    if target_shape is not None and mask.shape != target_shape:
        width = target_shape[1]
        height = target_shape[0]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32)
    cutoff = threshold if mask.max(initial=0.0) <= 1.0 else threshold * 255.0
    return mask > cutoff

def metric_values(prediction, target, smooth=1e-06):
    pred = prediction.astype(bool).ravel()
    gt = target.astype(bool).ravel()
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.float64)
    return {'dice': float((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)), 'iou': float((tp + smooth) / (tp + fp + fn + smooth)), 'precision': float((tp + smooth) / (tp + fp + smooth)), 'recall': float((tp + smooth) / (tp + fn + smooth))}

def surface_pixels(mask):
    return np.logical_xor(mask, binary_erosion(mask))

def boundary_metrics(prediction, target):
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

def evaluate_model(prediction_dir, samples, threshold, allow_missing):
    predictions = build_prediction_index(prediction_dir)
    totals = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'hd95': 0.0, 'assd': 0.0, 'boundary_f1': 0.0}
    missing = []
    evaluated = 0
    for image_path, mask_path in samples:
        prediction_path = predictions.get(image_path.stem)
        if prediction_path is None:
            missing.append(image_path.name)
            continue
        target = read_binary_mask(mask_path, threshold=0.5)
        prediction = read_binary_mask(prediction_path, threshold=threshold, target_shape=target.shape)
        metrics = metric_values(prediction, target)
        metrics.update(boundary_metrics(prediction, target))
        for key in totals:
            totals[key] += metrics[key]
        evaluated += 1
    if missing and (not allow_missing):
        raise FileNotFoundError(f'{prediction_dir.name}: {len(missing)} missing predictions')
    if evaluated == 0:
        raise ValueError(f'{prediction_dir.name}: no predictions evaluated')
    return {'model': MODEL_NAMES.get(prediction_dir.name, prediction_dir.name), 'model_key': prediction_dir.name, 'dice': totals['dice'] / evaluated, 'iou': totals['iou'] / evaluated, 'precision': totals['precision'] / evaluated, 'recall': totals['recall'] / evaluated, 'hd95': totals['hd95'] / evaluated, 'assd': totals['assd'] / evaluated, 'boundary_f1': totals['boundary_f1'] / evaluated, 'n_evaluated': evaluated, 'n_missing': len(missing)}

def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['model', 'model_key', 'dice', 'iou', 'precision', 'recall', 'hd95', 'assd', 'boundary_f1', 'n_evaluated', 'n_missing']
    with open(output_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'dice': f"{row['dice']:.6f}", 'iou': f"{row['iou']:.6f}", 'precision': f"{row['precision']:.6f}", 'recall': f"{row['recall']:.6f}", 'hd95': f"{row['hd95']:.6f}", 'assd': f"{row['assd']:.6f}", 'boundary_f1': f"{row['boundary_f1']:.6f}"})

def write_markdown(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['| Modelo | DICE | IoU | Precisao | Cobertura | HD95 | ASSD | BoundaryF1 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['dice']:.4f} | {row['iou']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['hd95']:.2f} | {row['assd']:.2f} | {row['boundary_f1']:.4f} |")
    output_path.write_text('\n'.join(lines) + '\n')

def main():
    args = parse_args()
    predictions_root = Path(args.predictions_root)
    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    output_dir = Path(args.output_dir)
    samples = load_samples(data_root, split_file, args.split, args.external)
    rows = [evaluate_model(path, samples, args.threshold, args.allow_missing) for path in model_dirs(predictions_root, args.models)]
    rows.sort(key=lambda row: row['dice'], reverse=True)
    csv_path = output_dir / f'{args.output_name}.csv'
    markdown_path = output_dir / f'{args.output_name}.md'
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    split_label = 'all' if args.external else args.split
    print(f'Evaluated {len(rows)} model(s) on {split_label} ({len(samples)} images).')
    print(f'CSV: {csv_path}')
    print(f'Markdown: {markdown_path}')
if __name__ == '__main__':
    main()
