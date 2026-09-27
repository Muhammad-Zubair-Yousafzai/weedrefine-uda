import pytest
import torch

from weedrefine.utils.mit_convert import convert, expected_mit_keys

DEPTHS, SR = (1, 1), (2, 1)   # tiny two-stage MiT: stage 1 has sr, stage 2 not


def _fake_hf(dim=4):
    sd = {"classifier.weight": torch.zeros(3, dim), "classifier.bias": torch.zeros(3)}
    e = "segformer.encoder."
    for s, sr in enumerate(SR):
        for p in ("weight", "bias"):
            sd[f"{e}patch_embeddings.{s}.proj.{p}"] = torch.zeros(dim)
            sd[f"{e}patch_embeddings.{s}.layer_norm.{p}"] = torch.zeros(dim)
            sd[f"{e}layer_norm.{s}.{p}"] = torch.zeros(dim)
            b = f"{e}block.{s}.0."
            mods = ["layer_norm_1", "attention.self.query", "attention.output.dense",
                    "layer_norm_2", "mlp.dense1", "mlp.dwconv.dwconv", "mlp.dense2"]
            if sr > 1:
                mods += ["attention.self.sr", "attention.self.layer_norm"]
            for m in mods:
                sd[f"{b}{m}.{p}"] = torch.zeros(dim)
            sd[f"{b}attention.self.key.{p}"] = torch.ones(dim)
            sd[f"{b}attention.self.value.{p}"] = 2 * torch.ones(dim)
    return sd


def test_convert_complete_and_kv_order():
    out = convert(_fake_hf(), DEPTHS, SR)
    assert set(out) == expected_mit_keys(DEPTHS, SR)
    kv = out["block1.0.attn.kv.weight"]
    assert kv.shape[0] == 8 and (kv[:4] == 1).all() and (kv[4:] == 2).all()  # key then value
    assert "block2.0.attn.sr.weight" not in out


def test_convert_fails_on_missing_weight():
    sd = _fake_hf()
    del sd["segformer.encoder.block.0.0.mlp.dense2.bias"]
    with pytest.raises(ValueError, match="missing"):
        convert(sd, DEPTHS, SR)


def test_convert_fails_on_unknown_weight():
    sd = _fake_hf()
    sd["segformer.encoder.block.0.0.attention.self.renamed.weight"] = torch.zeros(4)
    with pytest.raises(ValueError, match="unused"):
        convert(sd, DEPTHS, SR)


def test_expected_key_count_mit_b5():
    keys = expected_mit_keys((3, 6, 40, 3), (8, 4, 2, 1))
    # per block: 8 modules x 2 params, +4 with sr (stages 1-3); per stage: 6 embed/norm
    assert len(keys) == 52 * 16 + 49 * 4 + 4 * 6
