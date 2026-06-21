import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
TESTSET_GDRIVE_ID = '1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao'
ARCHIVE_NAME = 'polyp-testset.zip'
PREPROCESS_CONFIG = 'configs/preprocess_pc_endonorm.yaml'
IMAGE_SIZE = 352
SUBSET_DIRS = {'cvc-colondb': 'CVC-ColonDB', 'etis-larib': 'ETIS-LaribPolypDB', 'cvc-clinicdb': 'CVC-ClinicDB', 'cvc-300': 'CVC-300', 'kvasir': 'Kvasir'}
EXPECTED_SAMPLES = {'cvc-colondb': 380, 'etis-larib': 196, 'cvc-clinicdb': 62, 'cvc-300': 60, 'kvasir': 100}
DEFAULT_SUBSETS = ('cvc-colondb', 'etis-larib')

def parse_args():
    parser = argparse.ArgumentParser(description='Download and preprocess external polyp test datasets (PraNet TestDataset package).')
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--preprocessed-root', default='data/preprocessed')
    parser.add_argument('--subsets', nargs='*', default=list(DEFAULT_SUBSETS), choices=list(SUBSET_DIRS))
    parser.add_argument('--archive', default=None, help='Path to a pre-downloaded TestDataset zip (skips Google Drive).')
    parser.add_argument('--gdrive-id', default=TESTSET_GDRIVE_ID)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def run(command):
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)

def fetch_archive(gdrive_id, output_path):
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit('gdown is required to download from Google Drive. Install it (uv add gdown) or pass --archive with a pre-downloaded zip.') from exc
    gdown.download(id=gdrive_id, output=str(output_path), quiet=False)

def safe_extract(archive_path, output_dir):
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            target.relative_to(output_dir.resolve())
        archive.extractall(output_dir)

def find_subset_dir(root, subset_name):
    for candidate in [root, *root.rglob('*')]:
        if candidate.name == subset_name and (candidate / 'images').is_dir() and (candidate / 'masks').is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find '{subset_name}' with images/ and masks/ in the archive.")

def copy_dir(source, destination):
    destination.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        shutil.copy2(item, destination / item.name)

def install_subset(extracted_dir, subset_key, raw_root, force):
    source = find_subset_dir(extracted_dir, SUBSET_DIRS[subset_key])
    target = raw_root / subset_key
    if target.exists():
        if not force:
            raise FileExistsError(f'{target} already exists. Use --force to overwrite.')
        shutil.rmtree(target)
    copy_dir(source / 'images', target / 'images')
    copy_dir(source / 'masks', target / 'masks')
    return target

def raw_available(raw_root, subset_key):
    images = raw_root / subset_key / 'images'
    masks = raw_root / subset_key / 'masks'
    return images.is_dir() and masks.is_dir() and any(images.iterdir()) and any(masks.iterdir())

def preprocessed_available(preprocessed_root, subset_key):
    components = preprocessed_root / subset_key / 'images_norm'
    return components.is_dir() and any(components.iterdir())

def validate(target, subset_key):
    images = sorted((target / 'images').iterdir())
    masks = sorted((target / 'masks').iterdir())
    if len(images) != len(masks):
        raise ValueError(f'{subset_key}: {len(images)} images vs {len(masks)} masks.')
    expected = EXPECTED_SAMPLES.get(subset_key)
    if expected and len(images) != expected:
        print(f'WARNING: {subset_key} expected {expected} samples, found {len(images)}.')
    print(f'Validated {subset_key}: {len(images)} samples at {target}')

def download(args, raw_root):
    pending = [key for key in args.subsets if args.force or not raw_available(raw_root, key)]
    for key in args.subsets:
        if key not in pending:
            print(f'{key}: raw data already available at {raw_root / key}, skipping download.')
    if not pending:
        return
    with tempfile.TemporaryDirectory(prefix='polyp-testset-') as tmp:
        tmp_dir = Path(tmp)
        archive_path = Path(args.archive) if args.archive else tmp_dir / ARCHIVE_NAME
        if not args.archive:
            fetch_archive(args.gdrive_id, archive_path)
        extracted_dir = tmp_dir / 'extracted'
        extracted_dir.mkdir()
        safe_extract(archive_path, extracted_dir)
        for key in pending:
            target = install_subset(extracted_dir, key, raw_root, args.force)
            validate(target, key)

def preprocess(subset_key, raw_root, preprocessed_root):
    run(['uv', 'run', 'python', 'scripts/generate_pc_endonorm_dataset.py', '--dataset-name', subset_key, '--images-dir', str(raw_root / subset_key / 'images'), '--output-dir', str(preprocessed_root / subset_key), '--image-size', str(IMAGE_SIZE), '--config', PREPROCESS_CONFIG])

def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    preprocessed_root = Path(args.preprocessed_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    download(args, raw_root)
    for key in args.subsets:
        if args.force or not preprocessed_available(preprocessed_root, key):
            preprocess(key, raw_root, preprocessed_root)
        else:
            print(f'{key}: preprocessing already present at {preprocessed_root / key}, skipping.')
    print('\nCross datasets prepared.')
if __name__ == '__main__':
    main()
