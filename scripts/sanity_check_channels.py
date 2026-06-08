import argparse
from pathlib import Path

import torch

from models import get_model
from models.esfpnet import FIRST_CONV_WEIGHT_KEY, RGB_CHANNELS


BATCH_SIZE = 2
IMAGE_SIZE = 352
MODEL_TYPE = "b2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="weights/mit_b2.pth")
    return parser.parse_args()


def check_forward_backward(in_channels: int, pretrained_path: str | None) -> None:
    model = get_model(
        "esfpnet",
        num_classes=1,
        model_type=MODEL_TYPE,
        in_channels=in_channels,
        pretrained_path=pretrained_path,
    )
    x = torch.randn(BATCH_SIZE, in_channels, IMAGE_SIZE, IMAGE_SIZE)
    y = model(x)
    assert y.shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE), f"output shape {tuple(y.shape)}"

    y.mean().backward()
    first_conv = model.backbone.patch_embed1.proj
    assert first_conv.weight.shape[1] == in_channels
    assert first_conv.weight.grad is not None, "no gradient on first conv"

    print(f"  in_channels={in_channels} pretrained={pretrained_path is not None}: out {tuple(y.shape)}, grad OK")


def check_inflation(pretrained_path: str) -> None:
    source = torch.load(pretrained_path, map_location="cpu")[FIRST_CONV_WEIGHT_KEY]
    model = get_model(
        "esfpnet",
        num_classes=1,
        model_type=MODEL_TYPE,
        in_channels=8,
        pretrained_path=pretrained_path,
    )
    loaded = model.backbone.patch_embed1.proj.weight.detach()
    expected_extra = source.mean(dim=1, keepdim=True).repeat(1, 8 - RGB_CHANNELS, 1, 1)

    assert torch.allclose(loaded[:, :RGB_CHANNELS], source), "RGB channels not preserved"
    assert torch.allclose(loaded[:, RGB_CHANNELS:], expected_extra), "extra channels not mean-filled"
    print("  inflation OK: channels 0-2 = pretrained RGB, channels 3-7 = mean(RGB)")


def main() -> None:
    args = parse_args()
    has_pretrained = Path(args.pretrained).exists()
    pretrained = args.pretrained if has_pretrained else None

    print("Forward/backward checks:")
    check_forward_backward(3, None)
    check_forward_backward(8, None)
    if has_pretrained:
        check_forward_backward(3, pretrained)
        check_forward_backward(8, pretrained)

    print("Pretrained inflation check:")
    if has_pretrained:
        check_inflation(pretrained)
    else:
        print(f"  skipped (no pretrained weights at {args.pretrained})")

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
