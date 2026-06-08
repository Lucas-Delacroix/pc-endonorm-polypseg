from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Decoder import mlp
from Encoder import mit
from mmcv.cnn import ConvModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset


BACKBONES = {
    "B0": mit.mit_b0,
    "B1": mit.mit_b1,
    "B2": mit.mit_b2,
    "B3": mit.mit_b3,
    "B4": mit.mit_b4,
    "B5": mit.mit_b5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "export"))
    parser.add_argument("--model-type", default="B2", choices=sorted(BACKBONES))
    parser.add_argument("--train-root", default="../../data/kvasir/train")
    parser.add_argument("--val-root", default="../../data/kvasir/val")
    parser.add_argument("--data-root", default="../../../data/raw/kvasir-seg")
    parser.add_argument("--split-file", default="../../../data/splits/kvasir_split.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="../../../outputs/predictions/esfpnet")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class ESFPNetStructure(nn.Module):
    def __init__(self, model_type: str, pretrained_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.model_type = model_type
        self.backbone = BACKBONES[model_type]()

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        self.LP_1 = mlp.LP(input_dim=self.backbone.embed_dims[0], embed_dim=self.backbone.embed_dims[0])
        self.LP_2 = mlp.LP(input_dim=self.backbone.embed_dims[1], embed_dim=self.backbone.embed_dims[1])
        self.LP_3 = mlp.LP(input_dim=self.backbone.embed_dims[2], embed_dim=self.backbone.embed_dims[2])
        self.LP_4 = mlp.LP(input_dim=self.backbone.embed_dims[3], embed_dim=self.backbone.embed_dims[3])

        self.linear_fuse34 = ConvModule(
            in_channels=self.backbone.embed_dims[2] + self.backbone.embed_dims[3],
            out_channels=self.backbone.embed_dims[2],
            kernel_size=1,
            norm_cfg=dict(type="BN", requires_grad=True),
        )
        self.linear_fuse23 = ConvModule(
            in_channels=self.backbone.embed_dims[1] + self.backbone.embed_dims[2],
            out_channels=self.backbone.embed_dims[1],
            kernel_size=1,
            norm_cfg=dict(type="BN", requires_grad=True),
        )
        self.linear_fuse12 = ConvModule(
            in_channels=self.backbone.embed_dims[0] + self.backbone.embed_dims[1],
            out_channels=self.backbone.embed_dims[0],
            kernel_size=1,
            norm_cfg=dict(type="BN", requires_grad=True),
        )

        self.LP_12 = mlp.LP(input_dim=self.backbone.embed_dims[0], embed_dim=self.backbone.embed_dims[0])
        self.LP_23 = mlp.LP(input_dim=self.backbone.embed_dims[1], embed_dim=self.backbone.embed_dims[1])
        self.LP_34 = mlp.LP(input_dim=self.backbone.embed_dims[2], embed_dim=self.backbone.embed_dims[2])
        total_channels = sum(self.backbone.embed_dims)
        self.linear_pred = nn.Conv2d(total_channels, 1, kernel_size=1)

    def load_pretrained(self, pretrained_path: Union[str, Path]) -> None:
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
        if isinstance(pretrained_dict, dict) and "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        model_dict = self.backbone.state_dict()
        pretrained_dict = {key: value for key, value in pretrained_dict.items() if key in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]

        out_1, h, w = self.backbone.patch_embed1(x)
        for block in self.backbone.block1:
            out_1 = block(out_1, h, w)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        out_2, h, w = self.backbone.patch_embed2(out_1)
        for block in self.backbone.block2:
            out_2 = block(out_2, h, w)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        out_3, h, w = self.backbone.patch_embed3(out_2)
        for block in self.backbone.block3:
            out_3 = block(out_3, h, w)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        out_4, h, w = self.backbone.patch_embed4(out_3)
        for block in self.backbone.block4:
            out_4 = block(out_4, h, w)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([
            lp_3,
            F.interpolate(lp_4, scale_factor=2, mode="bilinear", align_corners=False),
        ], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([
            lp_2,
            F.interpolate(lp_34, scale_factor=2, mode="bilinear", align_corners=False),
        ], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([
            lp_1,
            F.interpolate(lp_23, scale_factor=2, mode="bilinear", align_corners=False),
        ], dim=1)))

        return self.linear_pred(torch.cat([
            lp_12,
            F.interpolate(lp_23, scale_factor=2, mode="bilinear", align_corners=False),
            F.interpolate(lp_34, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(lp_4, scale_factor=8, mode="bilinear", align_corners=False),
        ], dim=1))


_TRAIN_TRANSFORM = None
_VAL_TRANSFORM = None


def _get_train_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.GaussNoise(p=0.3),
        A.Equalize(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _get_val_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class PolypDataset(Dataset):
    def __init__(self, root: Union[str, Path], image_size: int, augmentations: bool):
        self.root = Path(root)
        self.images = sorted((self.root / "images").glob("*.jpg"))
        self.masks = [self.root / "masks" / image.name for image in self.images]
        self.transform = (
            _get_train_transform(image_size) if augmentations else _get_val_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks[index]).convert("L"))
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"].float()
        mask_t = augmented["mask"].float() / 255.0
        return image_t, mask_t.unsqueeze(0)


def split_samples(data_root: Path, split_file: Path, split: str) -> List[Path]:
    images = sorted((data_root / "images").glob("*.jpg"))
    with open(split_file) as file:
        indices = json.load(file)[split]
    return [images[index] for index in indices]


def val_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def structure_loss(pred: torch.Tensor, mask: torch.Tensor, smooth: float = 1) -> torch.Tensor:
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + smooth) / (union - inter + smooth)
    return (wbce + wiou).mean()


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-4) -> float:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return float((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def evaluate(model: nn.Module, root: Path, image_size: int, threshold: float, device: torch.device) -> float:
    dataset = PolypDataset(root, image_size, augmentations=False)
    transform = val_transform(image_size)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for image_path, mask_path in zip(dataset.images, dataset.masks):
            image = Image.open(image_path).convert("RGB")
            target = Image.open(mask_path).convert("L")
            target_array = np.asarray(target, np.float32)
            target_array /= target_array.max() + 1e-8
            tensor = transform(image).unsqueeze(0).to(device)
            pred = model(tensor)
            pred = F.interpolate(pred, size=target_array.shape, mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(pred) > threshold).float().cpu().squeeze()
            target_tensor = torch.from_numpy((target_array > 0.5).astype(np.float32))
            total += dice_score(pred, target_tensor)
    model.train()
    return total / len(dataset)


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return Path(f"SaveModel/ESFP_{args.model_type}_Endo_Kvasir/ESFPNet.pt")


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = ESFPNetStructure(args.model_type, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        PolypDataset(args.train_root, args.image_size, augmentations=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_dice = 0.0
    output_path = checkpoint_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        last_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            pred = model(images)
            pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = structure_loss(pred, masks)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        val_dice = evaluate(model, Path(args.val_root), args.image_size, args.threshold, device)
        print(f"Epoch [{epoch:5d}/{args.epochs:5d}] | loss: {last_loss:6.6f} | validation_coeffient: {val_dice:6.6f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": args.model_type,
                "image_size": args.image_size,
                "best_dice": best_dice,
            }, output_path)
            print(f"Saved {output_path}")


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint_path(args), map_location=device)
    if isinstance(state, nn.Module):
        return state.to(device).eval()
    model_type = state.get("model_type", args.model_type) if isinstance(state, dict) else args.model_type
    model = ESFPNetStructure(model_type).to(device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model.eval()


def save_mask(path: Path, mask: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.squeeze(mask)
    mask = ((mask > threshold) * 255).astype(np.uint8)
    cv2.imwrite(str(path), mask)


def export(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = load_model(args, device)
    transform = val_transform(args.image_size)
    output_dir = Path(args.output_dir)
    samples = split_samples(Path(args.data_root), Path(args.split_file), args.split)

    with torch.no_grad():
        for image_path in samples:
            image = Image.open(image_path).convert("RGB")
            shape = image.size[1], image.size[0]
            tensor = transform(image).unsqueeze(0).to(device)
            pred = model(tensor)
            pred = F.interpolate(pred, size=shape, mode="bilinear", align_corners=False)
            pred = torch.sigmoid(pred).cpu().numpy().squeeze()
            save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)

    print(f"Saved {len(samples)} predictions in: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.action == "train":
        train(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
