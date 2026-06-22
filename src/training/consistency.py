import torch
import torch.nn.functional as F
SOFT_IOU_EPS = 1e-06

def soft_iou(probability, target):
    intersection = (probability * target).sum()
    union = probability.sum() + target.sum() - intersection
    return (intersection + SOFT_IOU_EPS) / (union + SOFT_IOU_EPS)

def consistency_loss(student_probability, teacher_probability):
    return F.mse_loss(student_probability, teacher_probability)
