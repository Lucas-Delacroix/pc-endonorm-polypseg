import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / 'src'
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from data.datasets.multisource import build_source_dataset, normalized_name
from data.transforms.augmentation import get_val_transforms
from models import get_model
from scripts.train import load_config
from training.configured_datamodule import checkpoint_dir_for, is_multisource_config
DEFAULT_CONFIG = 'configs/experiments/MS_baseline_rgb.yaml'
DEFAULT_TEST_SOURCE = 'etis_larib'
DEFAULT_THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

def resolve_device(choice):
    if choice != 'auto':
        return choice
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def parse_thresholds(values):
    if not values:
        return list(DEFAULT_THRESHOLDS)
    thresholds = []
    for value in values:
        for token in value.replace(',', ' ').split():
            thresholds.append(float(token))
    return thresholds

def source_list(section):
    if 'sources' in section:
        return list(section['sources'])
    if 'source' in section:
        return [section['source']]
    return [section]

def select_source(config, role, source_name=None):
    dataset_config = config.get('dataset')
    if not isinstance(dataset_config, dict) or role not in dataset_config:
        raise ValueError(f'Config does not define dataset.{role}. Pass explicit images/masks paths.')
    sources = source_list(dataset_config[role])
    if source_name is None:
        if len(sources) != 1:
            names = ', '.join((source.get('name', '<unnamed>') for source in sources))
            raise ValueError(f'dataset.{role} has multiple sources ({names}); pass --source-name.')
        return dict(sources[0])
    wanted = normalized_name(source_name)
    for source in sources:
        if normalized_name(source.get('name', '')) == wanted:
            return dict(source)
    names = ', '.join((source.get('name', '<unnamed>') for source in sources))
    raise ValueError(f"Source '{source_name}' not found in dataset.{role}. Available: {names}")

def resolve_checkpoint(config, explicit):
    if explicit:
        return Path(explicit)
    return checkpoint_dir_for(config, is_multisource_config(config)) / 'best.pth'

def load_esfpnet_model(config, checkpoint_path, device):
    model_config = config['model']
    model = get_model(model_config['name'], num_classes=model_config['num_classes'], model_type=model_config.get('model_type', 'b2'), in_channels=model_config.get('in_channels', 3), pretrained_path=None)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get('model_state_dict', payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def build_dataset(config, *, role, images_dir=None, masks_dir=None, source_name=None, split='all', image_size=None):
    image_size = int(image_size or config['data']['image_size'])
    if images_dir or masks_dir:
        if not images_dir or not masks_dir:
            raise ValueError('Pass both images_dir and masks_dir, or neither.')
        source = {'name': source_name or role, 'images_dir': images_dir, 'masks_dir': masks_dir, 'split': split}
    else:
        config_role = 'test' if role == 'test' else 'val'
        source = select_source(config, config_role, source_name)
    dataset_config = config.get('dataset', {})
    dataset, split_description = build_source_dataset(source, transform=get_val_transforms(image_size), split_config=dataset_config.get('split', {}), base_dir=dataset_config.get('base_dir'))
    return (dataset, {'role': role, 'source': source, 'split_description': split_description, 'image_size': image_size})

def run_inference(model, dataset, *, device, batch_size, num_workers, pin_memory):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    logits_chunks = []
    probability_chunks = []
    target_chunks = []
    image_paths = []
    mask_paths = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['mask'].float()
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            probabilities = torch.sigmoid(logits)
            logits_chunks.append(logits[:, 0].detach().cpu().float().numpy())
            probability_chunks.append(probabilities[:, 0].detach().cpu().float().numpy())
            target_chunks.append((targets[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8))
            image_paths.extend((str(path) for path in batch.get('image_path', [])))
            mask_paths.extend((str(path) for path in batch.get('mask_path', [])))
    return {'logits': np.concatenate(logits_chunks, axis=0).astype(np.float32), 'probabilities': np.concatenate(probability_chunks, axis=0).astype(np.float32), 'targets': np.concatenate(target_chunks, axis=0).astype(np.uint8), 'image_paths': image_paths, 'mask_paths': mask_paths}

def collect_predictions(config, *, checkpoint, role, images_dir, masks_dir, source_name, split, image_size, batch_size, num_workers, device_choice):
    device = resolve_device(device_choice)
    checkpoint_path = resolve_checkpoint(config, checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    dataset, dataset_info = build_dataset(config, role=role, images_dir=images_dir, masks_dir=masks_dir, source_name=source_name, split=split, image_size=image_size)
    model = load_esfpnet_model(config, checkpoint_path, device)
    batch = int(batch_size or config['training']['batch_size'])
    workers = int(num_workers if num_workers is not None else config['data'].get('num_workers', 0))
    pin_memory = bool(config['data'].get('pin_memory', False) and device == 'cuda')
    predictions = run_inference(model, dataset, device=device, batch_size=batch, num_workers=workers, pin_memory=pin_memory)
    metadata = {'config_path': None, 'checkpoint': str(checkpoint_path), 'device': device, 'batch_size': batch, 'num_workers': workers, 'n_images': int(predictions['targets'].shape[0]), 'shape': list(predictions['targets'].shape), 'dataset': dataset_info}
    return (predictions, metadata)

def write_prediction_cache(path, predictions, metadata, *, overwrite=False):
    ensure_new_file(path, overwrite=overwrite)
    np.savez_compressed(path, logits=predictions['logits'], probabilities=predictions['probabilities'], targets=predictions['targets'], image_paths=np.asarray(predictions.get('image_paths', []), dtype=str), mask_paths=np.asarray(predictions.get('mask_paths', []), dtype=str), metadata=json.dumps(metadata, indent=2, sort_keys=True))
    return path

def read_prediction_cache(path):
    payload = np.load(path, allow_pickle=False)
    metadata_raw = payload['metadata'].item() if 'metadata' in payload else '{}'
    return {'logits': payload['logits'].astype(np.float32), 'probabilities': payload['probabilities'].astype(np.float32), 'targets': payload['targets'].astype(np.uint8), 'image_paths': payload['image_paths'].astype(str).tolist() if 'image_paths' in payload else [], 'mask_paths': payload['mask_paths'].astype(str).tolist() if 'mask_paths' in payload else [], 'metadata': json.loads(metadata_raw)}

def ensure_new_file(path, *, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not overwrite):
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')

def write_json(path, payload, *, overwrite=False):
    ensure_new_file(path, overwrite=overwrite)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + '\n')
    return path

def read_json(path):
    with open(path) as file:
        return json.load(file)

def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def unique_run_dir(root, run_name):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    base = root / f'{run_name}_{timestamp}'
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f'{base.name}_{suffix:02d}'
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
