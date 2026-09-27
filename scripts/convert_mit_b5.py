"""Download nvidia/mit-b5 (ImageNet-1K MiT-B5, Hugging Face) and save it as MIC's mit_b5.pth.

Replaces the SegFormer OneDrive/Google Drive downloads, which are dead (checked 2026-09-27).
The HF revision is pinned so the weights are reproducible.
"""
import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from weedrefine.utils.mit_convert import MIT_B5, convert

REVISION = "40357155205b036cf11b61f132d53d2f8861f170"   # nvidia/mit-b5 main, 2026-09-27


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/mit_b5.pth")
    args = ap.parse_args()

    path = hf_hub_download("nvidia/mit-b5", "pytorch_model.bin", revision=REVISION)
    hf = torch.load(path, map_location="cpu", weights_only=True)
    mit = convert(hf, **MIT_B5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # legacy format so MIC's torch 1.7.1 can read a file written by a new torch
    torch.save(mit, args.out, _use_new_zipfile_serialization=False)
    n = sum(t.numel() for t in mit.values())
    print(f"converted {len(mit)} tensors, {n / 1e6:.1f}M parameters -> {args.out}")
    print("kv block1.0:", tuple(mit["block1.0.attn.kv.weight"].shape),
          "| patch_embed1.proj:", tuple(mit["patch_embed1.proj.weight"].shape))


if __name__ == "__main__":
    main()
