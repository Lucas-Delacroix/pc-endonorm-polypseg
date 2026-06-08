import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)
    intersection = (pred * target).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def precision_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return ((tp + smooth) / (tp + fp + smooth)).item()


def recall_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return ((tp + smooth) / (tp + fn + smooth)).item()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "dice": dice_coefficient(pred, target),
        "iou": iou_score(pred, target),
        "precision": precision_score(pred, target),
        "recall": recall_score(pred, target),
    }