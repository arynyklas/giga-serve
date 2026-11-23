"""CPU tests run inside the image build: model math vs HF transformers, and the HTTP contract helpers.

  python test_giga_serve.py
"""
import base64
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import giga_model as G  # noqa: E402
import giga_serve as S  # noqa: E402

TINY = {
    "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 4,
    "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 24, "kv_lora_rank": 32,
    "intermediate_size": 96, "moe_intermediate_size": 32, "n_routed_experts": 8, "num_experts_per_tok": 2,
    "vocab_size": 100, "head_dim": 8, "first_k_dense_replace": 1, "n_shared_experts": 1, "n_group": 1,
    "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 1.0, "scoring_func": "sigmoid",
    "topk_method": "noaux_tc", "hidden_act": "silu", "rope_interleave": True, "is_causal": False,
    "rms_norm_eps": 1e-6, "rope_theta": 100000, "q_lora_rank": None, "moe_layer_freq": 1,
    "attention_bias": False, "max_position_embeddings": 64, "num_nextn_predict_layers": 0,
    "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 4.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                     "original_max_position_embeddings": 16, "rope_type": "yarn", "type": "yarn"},
}


def test_model_matches_hf():
    """Bidirectional DeepSeek-V3 forward + mean pool == transformers' DeepseekV3Model made non-causal."""
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model

    torch.manual_seed(0)
    c = dict(TINY, v_head_dim=24)
    c["v_head_dim"] = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]  # giga: v_head_dim == qk_head_dim
    hf_cfg = DeepseekV3Config(**{k: v for k, v in c.items() if k not in ("is_causal",)})
    hf_cfg._attn_implementation = "eager"
    hf = DeepseekV3Model(hf_cfg).eval().float()
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if p.ndim == 1 and "norm" in n:
                p.copy_(1 + 0.2 * torch.randn_like(p))
        for n, b in hf.named_buffers():
            if "e_score_correction_bias" in n:
                b.copy_(torch.randn_like(b) * 0.3)
    for layer in hf.layers:
        layer.self_attn.is_causal = False
    lens = [5, 9, 1, 3]
    ids = torch.randint(0, 100, (sum(lens),))
    refs = []
    with torch.no_grad():
        off = 0
        for n in lens:
            x = ids[off:off + n].unsqueeze(0)
            off += n
            h = hf.embed_tokens(x)
            pid = torch.arange(n).unsqueeze(0)
            pe = hf.rotary_emb(h, pid)
            for layer in hf.layers:
                h = layer(h, attention_mask=torch.zeros(1, 1, n, n), position_embeddings=pe, position_ids=pid,
                          past_key_values=None, use_cache=False, cache_position=torch.arange(n))
            refs.append(hf.norm(h)[0])
    cfg = G.Cfg(c)
    m = G.GigaEmbedModel(cfg, torch.device("cpu"), dtype=torch.float32, max_pos=64)
    sd = {n: t.detach().clone() for n, t in list(hf.named_parameters()) + list(hf.named_buffers())}
    layers = []
    for i in range(cfg.layers):
        def g(k):
            return sd[f"layers.{i}.{k}"]

        l = G.Layer()
        l.moe = i >= cfg.first_dense
        l.in_norm, l.post_norm = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        l.q, l.kv_a = g("self_attn.q_proj.weight"), g("self_attn.kv_a_proj_with_mqa.weight")
        l.kv_ln, l.kv_b = g("self_attn.kv_a_layernorm.weight"), g("self_attn.kv_b_proj.weight")
        l.o = g("self_attn.o_proj.weight")
        if l.moe:
            l.gate_w, l.gate_bias = g("mlp.gate.weight"), g("mlp.gate.e_score_correction_bias").float()
            l.w13, l.w2 = g("mlp.experts.gate_up_proj"), g("mlp.experts.down_proj")
            l.sh_gu = torch.cat([g("mlp.shared_experts.gate_proj.weight"), g("mlp.shared_experts.up_proj.weight")])
            l.sh_dn = g("mlp.shared_experts.down_proj.weight")
        else:
            l.gu = torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])
            l.dn = g("mlp.down_proj.weight")
        layers.append(l)
    m.layers, m.embed, m.norm_w = layers, sd["embed_tokens.weight"], sd["norm.weight"]
    cu = torch.tensor([0] + np.cumsum(lens).tolist(), dtype=torch.int32)
    seg = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
    pos = torch.arange(len(ids)) - cu[:-1].long()[seg]
    h = m.hidden(ids, pos, cu, max(lens))
    ref_h = torch.cat(refs)
    err = (h - ref_h).abs().max().item()
    assert err < 1e-4, err
    raw, emb = m.pool(h, seg, torch.tensor(lens, dtype=torch.float32), len(lens))
    ref = torch.stack([r.mean(0) for r in refs])
    assert torch.allclose(raw, ref, atol=1e-5)
    assert F.cosine_similarity(emb, F.normalize(ref, dim=-1)).min().item() > 0.999999


def test_parse():
    p = S.parse_embedding_request
    o = p({"input": "x"}, "m")
    assert o["texts"] == ["x"] and o["fmt"] == "float" and o["normalize"] and o["add_special"]
    assert p({"model": "m", "input": [1, 2]}, "m")["ids"] == [[1, 2]]
    assert p({"input": [[1], [2, 3]]}, "m")["ids"] == [[1], [2, 3]]
    assert p({"input": "x", "use_activation": False}, "m")["normalize"] is False
    for body, code, needle in [
        ({"model": "other", "input": "x"}, 404, "The model `other` does not exist."),
        ({"input": "x", "dimensions": 512}, 400,
         "does not support Matryoshka embeddings; dimensions must be unset (received dimensions=512)."),
        ({"input": "x", "normalize": True}, 400, "Parameter `normalize` was removed"),
        ({"input": []}, 400, "You must pass at least one prompt"),
        ({"input": ["a", 5]}, 400, "input must be"),
        ({"input": 5}, 400, "input must be"),
        ({"input": [True]}, 400, "input must be"),
        ({}, 400, "input"),
        ({"input": "x", "encoding_format": "hex"}, 400, "encoding_format"),
        ({"input": "x", "embed_dtype": "int8"}, 400, "embed_dtype"),
        ({"input": "x", "truncate_prompt_tokens": -2}, 400, "greater than or equal to -1"),
    ]:
        try:
            p(body, "m")
        except S.HTTPError as e:
            assert e.code == code and needle in e.message, (body, e.code, e.message)
            assert json.loads(e.body())["error"]["code"] == code
        else:
            raise AssertionError(body)


def test_truncate():
    t = S.truncate
    ids = [1, 10, 11, 12, 13]  # BOS + 4 content tokens
    assert t(ids, None, "right") == ids and t(ids, 9, "left") == ids
    assert t(ids, 3, "right") == [1, 10, 11]  # == HF truncation: BOS + first k-1
    assert t(ids, 3, "left") == [11, 12, 13]  # vLLM slices the encoded ids: BOS goes
    assert t(ids, 0, "right") == []


def test_encodings_and_bytes_contract():
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((3, 1536)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    raw = S.encode_vector(vecs[0], "float32", "native")
    assert np.array_equal(np.frombuffer(raw, "<f4"), vecs[0])
    assert np.array_equal(np.frombuffer(S.encode_vector(vecs[0], "float32", "big"), ">f4"), vecs[0])
    f16 = np.frombuffer(S.encode_vector(vecs[0], "float16", "little"), "<f2")
    assert np.array_equal(f16, vecs[0].astype(np.float16))
    bf = torch.from_numpy(vecs[0]).to(torch.bfloat16)
    big = np.frombuffer(S.encode_vector(vecs[0], "bfloat16", "big"), ">i2").astype("<i2")
    assert torch.equal(torch.from_numpy(big.copy()).view(torch.bfloat16), bf)
    fp8 = np.frombuffer(S.encode_vector(vecs[0], "fp8_e4m3", "native"), np.uint8)
    assert len(fp8) == 1536 and torch.equal(torch.from_numpy(fp8.copy()).view(torch.float8_e4m3fn).float(),
                                            torch.from_numpy(vecs[0]).to(torch.float8_e4m3fn).float())

    def opts(fmt, dtype="float32", endian="native"):
        return dict(fmt=fmt, dtype=dtype, endian=endian)

    st, hd, body = S.build_response(vecs, opts("float"), "embd-x", 1, "m", 7)
    j = json.loads(body)
    assert [d["index"] for d in j["data"]] == [0, 1, 2] and j["model"] == "m"
    assert np.array_equal(np.array(j["data"][1]["embedding"], dtype=np.float32), vecs[1])
    assert j["usage"] == {"prompt_tokens": 7, "total_tokens": 7, "completion_tokens": 0,
                          "prompt_tokens_details": None, "completion_tokens_details": None}
    st, hd, body = S.build_response(vecs, opts("base64"), "embd-x", 1, "m", 7)
    j = json.loads(body)
    assert np.array_equal(np.frombuffer(base64.b64decode(j["data"][2]["embedding"]), "<f4"), vecs[2])
    # vLLM's `bytes` contract: raw vectors in the body, a `metadata` header with contiguous items
    # from 0, shape [dim], end - start == dim * 4, usage last.
    st, hd, body = S.build_response(vecs, opts("bytes", "float32", "little"), "embd-x", 1, "m", 7)
    h = dict(hd)
    assert h[b"content-type"] == b"application/octet-stream"
    meta = json.loads(h[b"metadata"])
    off = 0
    for i, it in enumerate(meta["data"]):
        assert it["index"] == i and it["start"] == off and it["shape"] == [1536] and it["end"] - it["start"] == 6144
        assert it["embed_dtype"] == "float32" and it["endianness"] == "little"
        assert np.array_equal(np.frombuffer(body[it["start"]:it["end"]], "<f4"), vecs[i])
        off = it["end"]
    assert off == len(body) and meta["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
    st, hd, body = S.build_response(vecs, opts("bytes_only"), "embd-x", 1, "m", 7)
    assert b"metadata" not in dict(hd) and len(body) == 3 * 6144
    # build_response encodes the whole batch at once: must equal the per-row encodings, for every
    # dtype / endianness and for a non-contiguous row selection.
    for dtype in S.EMBED_DTYPES:
        for endian in S.ENDIANNESS:
            rows = vecs[::2]
            per_row = [S.encode_vector(np.ascontiguousarray(r), dtype, endian) for r in rows]
            st, hd, body = S.build_response(rows, opts("bytes", dtype, endian), "embd-x", 1, "m", 7)
            assert body == b"".join(per_row), (dtype, endian)
            j = json.loads(S.build_response(rows, opts("base64", dtype, endian), "embd-x", 1, "m", 7)[2])
            assert [base64.b64decode(d["embedding"]) for d in j["data"]] == per_row, (dtype, endian)


def test_too_long_message():
    e = S.too_long(2048)
    assert e.code == 400 and "maximum context length is 2048 tokens" in e.message and "value=2049" in e.message


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
