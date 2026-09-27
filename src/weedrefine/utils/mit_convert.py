"""Convert Hugging Face SegFormer encoder weights (nvidia/mit-bX) to the original MiT names.

MIC's mix_transformer.py loads pretrained/mit_b5.pth with strict=False, so a naming
mismatch silently leaves the backbone random. convert() therefore fails loudly unless
every HF encoder weight is used and every expected MiT weight is produced.
Name mapping follows transformers' convert_segformer_original_to_pytorch.py in reverse.
"""
import re

import torch

MIT_B5 = dict(depths=(3, 6, 40, 3), sr_ratios=(8, 4, 2, 1))

_BLOCK_MAP = {
    "layer_norm_1": "norm1",
    "layer_norm_2": "norm2",
    "attention.self.query": "attn.q",
    "attention.self.sr": "attn.sr",
    "attention.self.layer_norm": "attn.norm",
    "attention.output.dense": "attn.proj",
    "mlp.dense1": "mlp.fc1",
    "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
    "mlp.dense2": "mlp.fc2",
}


def expected_mit_keys(depths, sr_ratios):
    keys = set()
    for s, (depth, sr) in enumerate(zip(depths, sr_ratios), start=1):
        for p in ("weight", "bias"):
            keys |= {f"patch_embed{s}.proj.{p}", f"patch_embed{s}.norm.{p}", f"norm{s}.{p}"}
            for j in range(depth):
                parts = ["norm1", "attn.q", "attn.kv", "attn.proj", "norm2",
                         "mlp.fc1", "mlp.dwconv.dwconv", "mlp.fc2"]
                if sr > 1:
                    parts += ["attn.sr", "attn.norm"]
                keys |= {f"block{s}.{j}.{m}.{p}" for m in parts}
    return keys


def convert(hf_state, depths, sr_ratios):
    """HF state dict -> MiT state dict. Raises ValueError listing any unmatched names."""
    out, used, kv = {}, set(), {}
    for name, t in hf_state.items():
        n = name[len("segformer."):] if name.startswith("segformer.") else name
        if not n.startswith("encoder."):
            continue  # classifier head, not part of the backbone
        n = n[len("encoder."):]
        if m := re.fullmatch(r"patch_embeddings\.(\d+)\.(proj|layer_norm)\.(weight|bias)", n):
            s, mod, p = int(m[1]) + 1, {"proj": "proj", "layer_norm": "norm"}[m[2]], m[3]
            out[f"patch_embed{s}.{mod}.{p}"] = t
        elif m := re.fullmatch(r"layer_norm\.(\d+)\.(weight|bias)", n):
            out[f"norm{int(m[1]) + 1}.{m[2]}"] = t
        elif m := re.fullmatch(r"block\.(\d+)\.(\d+)\.(.+)\.(weight|bias)", n):
            s, j, mod, p = int(m[1]) + 1, int(m[2]), m[3], m[4]
            if mod in ("attention.self.key", "attention.self.value"):
                kv.setdefault((s, j, p), {})[mod.rsplit(".", 1)[1]] = t
            elif mod in _BLOCK_MAP:
                out[f"block{s}.{j}.{_BLOCK_MAP[mod]}.{p}"] = t
            else:
                continue
        else:
            continue
        used.add(name)
    for (s, j, p), d in kv.items():
        if set(d) != {"key", "value"}:
            raise ValueError(f"block{s}.{j}: key/value {p} incomplete")
        out[f"block{s}.{j}.attn.kv.{p}"] = torch.cat([d["key"], d["value"]], 0)

    unused = sorted(n for n in hf_state if "encoder." in n and n not in used)
    expected = expected_mit_keys(depths, sr_ratios)
    missing, extra = sorted(expected - set(out)), sorted(set(out) - expected)
    if unused or missing or extra:
        raise ValueError(f"unused HF encoder weights {unused[:10]} ({len(unused)}), "
                         f"missing MiT weights {missing[:10]} ({len(missing)}), "
                         f"unexpected {extra[:10]} ({len(extra)})")
    return out
