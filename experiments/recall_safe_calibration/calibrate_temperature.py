from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.recall_safe_calibration.metrics_calibration import compute_metrics  # noqa: E402
from experiments.recall_safe_calibration.utils import (  # noqa: E402
    DEFAULT_CONFIG,
    collect_predictions,
    load_config,
    read_prediction_cache,
    resolve_device,
    write_json,
    write_prediction_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit scalar Temperature Scaling on validation logits for binary segmentation."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache", default=None, help="Optional .npz cache from collect_logits.py.")
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--val-masks", default=None)
    parser.add_argument("--val-source-name", default=None)
    parser.add_argument("--val-split", default="all")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-device", default="cpu")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--lesion-dilation", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_or_collect_validation(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.cache:
        return read_prediction_cache(Path(args.cache))

    predictions, metadata = collect_predictions(
        config,
        checkpoint=args.checkpoint,
        role="val",
        images_dir=args.val_images,
        masks_dir=args.val_masks,
        source_name=args.val_source_name,
        split=args.val_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_choice=args.device,
    )
    metadata["config_path"] = args.config
    if args.cache_output:
        write_prediction_cache(Path(args.cache_output), predictions, metadata, overwrite=args.overwrite)
    return {**predictions, "metadata": metadata}


def fit_temperature(
    logits,
    targets,
    *,
    device: str,
    max_iter: int,
    lr: float,
) -> dict[str, float]:
    calibration_device = resolve_device(device) if device == "auto" else device
    logits_tensor = torch.as_tensor(logits.reshape(-1), dtype=torch.float32, device=calibration_device)
    targets_tensor = torch.as_tensor(targets.reshape(-1), dtype=torch.float32, device=calibration_device)
    log_temperature = torch.zeros((), dtype=torch.float32, device=calibration_device, requires_grad=True)

    with torch.no_grad():
        initial_loss = F.binary_cross_entropy_with_logits(logits_tensor, targets_tensor).item()

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = F.binary_cross_entropy_with_logits(logits_tensor / temperature, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        final_loss = F.binary_cross_entropy_with_logits(logits_tensor / temperature, targets_tensor).item()

    return {
        "temperature": float(temperature.detach().cpu().item()),
        "log_temperature": float(log_temperature.detach().cpu().item()),
        "initial_nll": float(initial_loss),
        "final_nll": float(final_loss),
    }


def temperature_payload(
    cache: dict[str, Any],
    fit: dict[str, float],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    before = compute_metrics(
        cache["logits"],
        cache["targets"],
        threshold=0.5,
        temperature=1.0,
        ece_bins=args.ece_bins,
        lesion_dilation=args.lesion_dilation,
        include_boundary=False,
    )
    after = compute_metrics(
        cache["logits"],
        cache["targets"],
        threshold=0.5,
        temperature=fit["temperature"],
        ece_bins=args.ece_bins,
        lesion_dilation=args.lesion_dilation,
        include_boundary=False,
    )
    return {
        "method": "temperature_scaling",
        "objective": "BCEWithLogitsLoss(logits / exp(log_T), target)",
        "temperature": fit["temperature"],
        "log_temperature": fit["log_temperature"],
        "optimization": {
            "parameterization": "T = exp(log_T), clamped to [1e-3, 100]",
            "optimizer": "LBFGS",
            "lr": args.lr,
            "max_iter": args.max_iter,
            "initial_T": 1.0,
            "initial_nll": fit["initial_nll"],
            "final_nll": fit["final_nll"],
        },
        "validation_cache_metadata": cache.get("metadata", {}),
        "validation_metrics_at_threshold_0_5": {
            "before_temperature_scaling": before,
            "after_temperature_scaling": after,
        },
        "config": {
            "config_path": args.config,
            "checkpoint": args.checkpoint,
            "ece_bins": args.ece_bins,
            "lesion_dilation": args.lesion_dilation,
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cache = load_or_collect_validation(args, config)
    fit = fit_temperature(
        cache["logits"],
        cache["targets"],
        device=args.calibration_device,
        max_iter=args.max_iter,
        lr=args.lr,
    )
    payload = temperature_payload(cache, fit, args=args)
    write_json(Path(args.output), payload, overwrite=args.overwrite)
    print(f"Temperature T={fit['temperature']:.6f} saved to: {args.output}")


if __name__ == "__main__":
    main()
