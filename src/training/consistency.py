from __future__ import annotations

import torch
import torch.nn.functional as F

SOFT_IOU_EPS = 1e-6


def soft_iou(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).sum()
    union = probability.sum() + target.sum() - intersection
    return (intersection + SOFT_IOU_EPS) / (union + SOFT_IOU_EPS)


def consistency_loss(student_probability: torch.Tensor, teacher_probability: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_probability, teacher_probability)
