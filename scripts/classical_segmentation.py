import argparse
from pathlib import Path
import cv2
from segmentation.classical import segment
try:
    from scripts.evaluate_predictions import load_samples
except ModuleNotFoundError:
    from evaluate_predictions import load_samples
DEFAULT_DATA_ROOT = 'data/raw/kvasir-seg'
DEFAULT_SPLIT_FILE = 'data/splits/kvasir_split.json'
DEFAULT_OUTPUT_ROOT = 'outputs/predictions'

def parse_args():
    parser = argparse.ArgumentParser(description='Classical (training-free) PDI polyp segmentation as a comparison baseline.')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split-file', default=DEFAULT_SPLIT_FILE)
    parser.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--name', default=None)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()

def main():
    args = parse_args()
    name = args.name or 'Classical_Otsu'
    output_dir = Path(args.output_root) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(Path(args.data_root), Path(args.split_file), args.split, args.external)
    for image_path, _ in samples:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f'Invalid image: {image_path}')
        mask = segment(bgr)
        cv2.imwrite(str(output_dir / f'{Path(image_path).stem}.png'), mask)
    print(f'Wrote {len(samples)} {name} masks to: {output_dir}')
if __name__ == '__main__':
    main()
