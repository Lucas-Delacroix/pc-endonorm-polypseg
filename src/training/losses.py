import torch
import torch.nn as nn
import torch.nn.functional as F

DICE_SMOOTH = 1e-6
STRUCTURE_SMOOTH = 1.0
EDGE_WEIGHT = 5
EDGE_KERNEL = 15
EDGE_PADDING = 7


class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).contiguous().view(-1)
        target = target.contiguous().view(-1)
        overlap = (pred * target).sum()
        score = (2.0 * overlap + DICE_SMOOTH) / (pred.sum() + target.sum() + DICE_SMOOTH)
        return 1.0 - score


class StructureLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge = F.avg_pool2d(target, kernel_size=EDGE_KERNEL, stride=1, padding=EDGE_PADDING) - target
        weight = 1.0 + EDGE_WEIGHT * torch.abs(edge)

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        prob = torch.sigmoid(pred)
        overlap = (prob * target * weight).sum(dim=(2, 3))
        union = ((prob + target) * weight).sum(dim=(2, 3))
        wiou = 1.0 - (overlap + STRUCTURE_SMOOTH) / (union - overlap + STRUCTURE_SMOOTH)
        return (bce + wiou).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).contiguous().view(-1)
        target = target.contiguous().view(-1)

        tp = (pred * target).sum()
        fp = (pred * (1.0 - target)).sum()
        fn = ((1.0 - pred) * target).sum()
        score = (tp + DICE_SMOOTH) / (tp + self.alpha * fp + self.beta * fn + DICE_SMOOTH)
        return torch.pow(1.0 - score, self.gamma)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.4,
        bce_weight: float = 0.2,
        focal_tversky_weight: float = 0.4,
        focal_tversky: dict | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_tversky = FocalTverskyLoss(**(focal_tversky or {}))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(pred, target)
            + self.bce_weight * self.bce(pred, target)
            + self.focal_tversky_weight * self.focal_tversky(pred, target)
        )


def build_loss(config) -> nn.Module:
    if config == "structure":
        return StructureLoss()

    config = dict(config)
    name = config.pop("name")
    if name == "combined":
        return CombinedLoss(**config)
    raise ValueError(f"Unknown loss: {name}")
