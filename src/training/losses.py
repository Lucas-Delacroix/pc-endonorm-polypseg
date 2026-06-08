import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        loss = 1 - (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class StructureLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=15, stride=1, padding=7) - target
        )
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        pred_sigmoid = torch.sigmoid(pred)
        inter = (pred_sigmoid * target * weight).sum(dim=(2, 3))
        union = ((pred_sigmoid + target) * weight).sum(dim=(2, 3))
        weighted_iou = 1 - (inter + self.smooth) / (union - inter + self.smooth)

        loss = (weighted_bce + weighted_iou).mean()
        return loss


LOSSES = {
    "bce_dice": BCEDiceLoss,
    "dice": DiceLoss,
    "structure": StructureLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    assert name in LOSSES, (
        f"Loss '{name}' não reconhecida. Opções: {list(LOSSES.keys())}"
    )
    return LOSSES[name](**kwargs)