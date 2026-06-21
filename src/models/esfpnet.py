from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models import register_model
from models.base_model import BaseModel


MIT_CONFIGS = {
    "b2": {
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 4, 6, 3],
    },
}

MIT_LAYERNORM_EPS = 1e-6
RGB_CHANNELS = 3
FIRST_CONV_WEIGHT_KEY = "patch_embed1.proj.weight"


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class DepthwiseConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).view(batch_size, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthwiseConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, tokens, channels = x.shape
        q = self.q(x).reshape(
            batch_size,
            tokens,
            self.num_heads,
            channels // self.num_heads,
        )
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_for_kv = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
            x_for_kv = self.sr(x_for_kv).reshape(batch_size, channels, -1)
            x_for_kv = x_for_kv.permute(0, 2, 1)
            x_for_kv = self.norm(x_for_kv)
        else:
            x_for_kv = x

        kv = self.kv(x_for_kv).reshape(
            batch_size,
            -1,
            2,
            self.num_heads,
            channels // self.num_heads,
        )
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attn = (q @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(batch_size, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=MIT_LAYERNORM_EPS)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=MIT_LAYERNORM_EPS)
        self.mlp = MixFeedForward(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: list[int],
        num_heads: list[int],
        depths: list[int],
        mlp_ratios: list[float] | None = None,
        sr_ratios: list[int] | None = None,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        mlp_ratios = mlp_ratios or [4.0, 4.0, 4.0, 4.0]
        sr_ratios = sr_ratios or [8, 4, 2, 1]

        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_channels, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        block_start = 0
        self.block1 = self._make_stage(
            embed_dims[0],
            num_heads[0],
            depths[0],
            mlp_ratios[0],
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            dpr[block_start:block_start + depths[0]],
            sr_ratios[0],
        )
        block_start += depths[0]
        self.block2 = self._make_stage(
            embed_dims[1],
            num_heads[1],
            depths[1],
            mlp_ratios[1],
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            dpr[block_start:block_start + depths[1]],
            sr_ratios[1],
        )
        block_start += depths[1]
        self.block3 = self._make_stage(
            embed_dims[2],
            num_heads[2],
            depths[2],
            mlp_ratios[2],
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            dpr[block_start:block_start + depths[2]],
            sr_ratios[2],
        )
        block_start += depths[2]
        self.block4 = self._make_stage(
            embed_dims[3],
            num_heads[3],
            depths[3],
            mlp_ratios[3],
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            dpr[block_start:block_start + depths[3]],
            sr_ratios[3],
        )

        self.norm1 = nn.LayerNorm(embed_dims[0], eps=MIT_LAYERNORM_EPS)
        self.norm2 = nn.LayerNorm(embed_dims[1], eps=MIT_LAYERNORM_EPS)
        self.norm3 = nn.LayerNorm(embed_dims[2], eps=MIT_LAYERNORM_EPS)
        self.norm4 = nn.LayerNorm(embed_dims[3], eps=MIT_LAYERNORM_EPS)

        self.apply(self._init_weights)

    @staticmethod
    def _make_stage(
        dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rates: list[float],
        sr_ratio: int,
    ) -> nn.ModuleList:
        return nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                sr_ratio=sr_ratio,
            )
            for i in range(depth)
        ])

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _tokens_to_feature_map(
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, _, channels = x.shape
        return x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []

        x, height, width = self.patch_embed1(x)
        for block in self.block1:
            x = block(x, height, width)
        x = self.norm1(x)
        x = self._tokens_to_feature_map(x, height, width)
        features.append(x)

        x, height, width = self.patch_embed2(x)
        for block in self.block2:
            x = block(x, height, width)
        x = self.norm2(x)
        x = self._tokens_to_feature_map(x, height, width)
        features.append(x)

        x, height, width = self.patch_embed3(x)
        for block in self.block3:
            x = block(x, height, width)
        x = self.norm3(x)
        x = self._tokens_to_feature_map(x, height, width)
        features.append(x)

        x, height, width = self.patch_embed4(x)
        for block in self.block4:
            x = block(x, height, width)
        x = self.norm4(x)
        x = self._tokens_to_feature_map(x, height, width)
        features.append(x)

        return features


class LinearProjection(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ESFPDecoder(nn.Module):
    def __init__(self, embed_dims: list[int], num_classes: int):
        super().__init__()
        dim1, dim2, dim3, dim4 = embed_dims

        self.lp_1 = LinearProjection(dim1, dim1)
        self.lp_2 = LinearProjection(dim2, dim2)
        self.lp_3 = LinearProjection(dim3, dim3)
        self.lp_4 = LinearProjection(dim4, dim4)

        self.linear_fuse34 = ConvBNAct(dim3 + dim4, dim3)
        self.linear_fuse23 = ConvBNAct(dim2 + dim3, dim2)
        self.linear_fuse12 = ConvBNAct(dim1 + dim2, dim1)

        self.lp_34 = LinearProjection(dim3, dim3)
        self.lp_23 = LinearProjection(dim2, dim2)
        self.lp_12 = LinearProjection(dim1, dim1)

        self.linear_pred = nn.Conv2d(dim1 + dim2 + dim3 + dim4, num_classes, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        out_1, out_2, out_3, out_4 = features

        lp_1 = self.lp_1(out_1)
        lp_2 = self.lp_2(out_2)
        lp_3 = self.lp_3(out_3)
        lp_4 = self.lp_4(out_4)

        lp_34 = self.linear_fuse34(torch.cat([
            lp_3,
            F.interpolate(lp_4, size=lp_3.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))
        lp_34 = self.lp_34(lp_34)

        lp_23 = self.linear_fuse23(torch.cat([
            lp_2,
            F.interpolate(lp_34, size=lp_2.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))
        lp_23 = self.lp_23(lp_23)

        lp_12 = self.linear_fuse12(torch.cat([
            lp_1,
            F.interpolate(lp_23, size=lp_1.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))
        lp_12 = self.lp_12(lp_12)

        target_size = lp_12.shape[-2:]
        out = self.linear_pred(torch.cat([
            lp_12,
            F.interpolate(lp_23, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(lp_34, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(lp_4, size=target_size, mode="bilinear", align_corners=False),
        ], dim=1))

        return F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)


@register_model("esfpnet")
class ESFPNet(BaseModel):
    def __init__(
        self,
        num_classes: int = 1,
        model_type: str = "b2",
        in_channels: int = 3,
        pretrained_path: str | Path | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__(num_classes=num_classes)
        model_type = model_type.lower()
        if model_type not in MIT_CONFIGS:
            raise ValueError(
                f"Unknown ESFPNet model_type '{model_type}'. "
                f"Options: {list(MIT_CONFIGS.keys())}"
            )

        self.model_type = model_type
        cfg = MIT_CONFIGS[model_type]
        self.backbone = MixVisionTransformer(
            in_channels=in_channels,
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            depths=cfg["depths"],
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = ESFPDecoder(cfg["embed_dims"], num_classes=num_classes)

        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        output_size = x.shape[-2:]
        features = self.backbone(x)
        output = self.decoder(features, output_size)
        if return_features:
            return output, features
        return output

    def load_pretrained_encoder(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_prefixes(state_dict)

        backbone_state = self.backbone.state_dict()
        if FIRST_CONV_WEIGHT_KEY in state_dict and FIRST_CONV_WEIGHT_KEY in backbone_state:
            state_dict[FIRST_CONV_WEIGHT_KEY] = self._inflate_first_conv(
                state_dict[FIRST_CONV_WEIGHT_KEY],
                backbone_state[FIRST_CONV_WEIGHT_KEY].shape[1],
            )

        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in backbone_state and backbone_state[key].shape == value.shape
        }
        missing = sorted(set(backbone_state) - set(compatible))
        self.backbone.load_state_dict(compatible, strict=False)

        if not compatible:
            raise ValueError(f"No compatible encoder weights found in {checkpoint_path}.")

        print(
            f"Loaded {len(compatible)} MiT encoder tensors from {checkpoint_path}. "
            f"Missing tensors: {len(missing)}"
        )

    @staticmethod
    def _inflate_first_conv(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
        source_in_channels = weight.shape[1]
        if source_in_channels == target_in_channels:
            return weight
        if target_in_channels < source_in_channels:
            return weight[:, :target_in_channels]
        mean_weight = weight.mean(dim=1, keepdim=True)
        extra = mean_weight.repeat(1, target_in_channels - source_in_channels, 1, 1)
        return torch.cat([weight, extra], dim=1)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
            return checkpoint
        raise TypeError("Checkpoint must be a state-dict or a dict containing one.")

    @staticmethod
    def _strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cleaned = {}
        prefixes = ("module.", "backbone.")
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value
        return cleaned
