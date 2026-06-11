from __future__ import annotations

import torch
import torch.nn.functional as F

DINO_PATCH_SIZE = 14
DINO_REPO = "facebookresearch/dinov2"


class DinoTeacher:
    def __init__(self, model_name: str = "dinov2_vits14", image_size: int = 350, device: str = "cpu"):
        self.device = device
        self.image_size = image_size - (image_size % DINO_PATCH_SIZE)
        self.grid = self.image_size // DINO_PATCH_SIZE
        self.model = torch.hub.load(DINO_REPO, model_name).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.embed_dim = self.model.embed_dim

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        tokens = self.model.forward_features(resized)["x_norm_patchtokens"]
        feature_map = tokens.reshape(images.shape[0], self.grid, self.grid, self.embed_dim)
        return feature_map.permute(0, 3, 1, 2).contiguous()


def feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student, dim=1)
    teacher = F.normalize(teacher, dim=1)
    cosine = (student * teacher).sum(dim=1)
    return (1.0 - cosine).mean()
