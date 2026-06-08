import argparse
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import torch


MIT_DEPTHS = {
    "b0": [2, 2, 2, 2],
    "b1": [2, 2, 2, 2],
    "b2": [3, 4, 6, 3],
    "b3": [3, 4, 18, 3],
    "b4": [3, 8, 27, 3],
    "b5": [3, 6, 40, 3],
}


DEFAULT_HF_URL = "https://huggingface.co/nvidia/mit-{model_type}/resolve/main/pytorch_model.bin"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="b2",
        choices=sorted(MIT_DEPTHS.keys()),
        help="MiT variant to download.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pth path. Defaults to weights/mit_<model_type>.pth.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Override download URL.",
    )
    parser.add_argument(
        "--checkpoint_format",
        choices=("hf", "mit"),
        default="hf",
        help="Use 'hf' for Hugging Face SegFormer weights or 'mit' for native MiT keys.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "colonoscopy-segmentation-reproduction"})
    with urlopen(request) as response, open(destination, "wb") as output_file:
        shutil.copyfileobj(response, output_file)


def extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError("Checkpoint must be a state-dict or a dict containing one.")


def strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "backbone.", "encoder.")
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def convert_hf_segformer_to_mit(
    state_dict: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, torch.Tensor]:
    depths = MIT_DEPTHS[model_type]
    converted = {}

    def copy(source: str, target: str) -> None:
        if source in state_dict:
            converted[target] = state_dict[source]

    def concat(source_a: str, source_b: str, target: str) -> None:
        if source_a in state_dict and source_b in state_dict:
            converted[target] = torch.cat([state_dict[source_a], state_dict[source_b]], dim=0)

    for stage_idx, depth in enumerate(depths, start=1):
        hf_stage = stage_idx - 1
        hf_patch = f"segformer.encoder.patch_embeddings.{hf_stage}"
        mit_patch = f"patch_embed{stage_idx}"

        copy(f"{hf_patch}.proj.weight", f"{mit_patch}.proj.weight")
        copy(f"{hf_patch}.proj.bias", f"{mit_patch}.proj.bias")
        copy(f"{hf_patch}.layer_norm.weight", f"{mit_patch}.norm.weight")
        copy(f"{hf_patch}.layer_norm.bias", f"{mit_patch}.norm.bias")
        copy(f"segformer.encoder.layer_norm.{hf_stage}.weight", f"norm{stage_idx}.weight")
        copy(f"segformer.encoder.layer_norm.{hf_stage}.bias", f"norm{stage_idx}.bias")

        for block_idx in range(depth):
            hf_block = f"segformer.encoder.block.{hf_stage}.{block_idx}"
            mit_block = f"block{stage_idx}.{block_idx}"

            copy(f"{hf_block}.layer_norm_1.weight", f"{mit_block}.norm1.weight")
            copy(f"{hf_block}.layer_norm_1.bias", f"{mit_block}.norm1.bias")
            copy(f"{hf_block}.layer_norm_2.weight", f"{mit_block}.norm2.weight")
            copy(f"{hf_block}.layer_norm_2.bias", f"{mit_block}.norm2.bias")

            copy(f"{hf_block}.attention.self.query.weight", f"{mit_block}.attn.q.weight")
            copy(f"{hf_block}.attention.self.query.bias", f"{mit_block}.attn.q.bias")
            concat(
                f"{hf_block}.attention.self.key.weight",
                f"{hf_block}.attention.self.value.weight",
                f"{mit_block}.attn.kv.weight",
            )
            concat(
                f"{hf_block}.attention.self.key.bias",
                f"{hf_block}.attention.self.value.bias",
                f"{mit_block}.attn.kv.bias",
            )
            copy(f"{hf_block}.attention.output.dense.weight", f"{mit_block}.attn.proj.weight")
            copy(f"{hf_block}.attention.output.dense.bias", f"{mit_block}.attn.proj.bias")
            copy(f"{hf_block}.attention.self.sr.weight", f"{mit_block}.attn.sr.weight")
            copy(f"{hf_block}.attention.self.sr.bias", f"{mit_block}.attn.sr.bias")
            copy(f"{hf_block}.attention.self.layer_norm.weight", f"{mit_block}.attn.norm.weight")
            copy(f"{hf_block}.attention.self.layer_norm.bias", f"{mit_block}.attn.norm.bias")

            copy(f"{hf_block}.mlp.dense1.weight", f"{mit_block}.mlp.fc1.weight")
            copy(f"{hf_block}.mlp.dense1.bias", f"{mit_block}.mlp.fc1.bias")
            copy(f"{hf_block}.mlp.dwconv.dwconv.weight", f"{mit_block}.mlp.dwconv.dwconv.weight")
            copy(f"{hf_block}.mlp.dwconv.dwconv.bias", f"{mit_block}.mlp.dwconv.dwconv.bias")
            copy(f"{hf_block}.mlp.dense2.weight", f"{mit_block}.mlp.fc2.weight")
            copy(f"{hf_block}.mlp.dense2.bias", f"{mit_block}.mlp.fc2.bias")

    if not converted:
        raise ValueError("No Hugging Face SegFormer tensors were converted.")

    return converted


def main():
    args = parse_args()
    output_path = Path(args.output or f"weights/mit_{args.model_type}.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = args.url or DEFAULT_HF_URL.format(model_type=args.model_type.replace("_", "-"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = Path(tmp_dir) / "pytorch_model.bin"
        print(f"Downloading {url}")
        download_file(url, raw_path)

        checkpoint = torch.load(raw_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        if args.checkpoint_format == "hf":
            state_dict = convert_hf_segformer_to_mit(state_dict, args.model_type)
        else:
            state_dict = strip_prefixes(state_dict)

    torch.save(state_dict, output_path)
    print(f"Saved {len(state_dict)} tensors to {output_path}")


if __name__ == "__main__":
    main()
