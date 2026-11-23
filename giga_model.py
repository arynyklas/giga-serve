"""Giga-Embeddings-instruct-10B-A1.8B: bidirectional DeepSeek-V3 (MLA weights, MHA compute) + MoE.

Forward over a flat token stream with cu_seqlens, no KV cache; output is the MEAN pool over every
prompt token (BOS included) in fp32, L2-normalised -- the vLLM `--pooler-config
{"pooling_type":"MEAN","use_activation":true}` recipe. The op sequence follows
vllm/model_executor/models/deepseek_v2.py (DeepseekV2Attention, the non-MLA ENCODER_ONLY path)
and uses the same kernels as vLLM v0.34 in eager mode: cuBLAS GEMMs with vLLM's shapes, vLLM
csrc rms_norm / fused_add_rms_norm / silu_and_mul / grouped_topk, FlashInfer RoPE on an fp32
cos/sin cache, FA3 varlen non-causal, and vLLM's Triton fused_moe with its tuned H200 config.
Without CUDA (unit tests) every op falls back to plain torch.
"""
import glob
import json
import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as F

try:
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    HAVE_CUDA_OPS, _CUDA_OPS_ERR = True, None
except Exception as _e:  # CPU unit tests
    HAVE_CUDA_OPS, _CUDA_OPS_ERR = False, _e

Tensor = torch.Tensor


def yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


class Cfg:
    """The subset of config.json the forward needs; anything this code does not implement is refused."""

    hidden: int
    layers: int
    heads: int
    nope: int
    rope: int
    vdim: int
    qk: int
    kv_lora: int
    inter: int
    moe_inter: int
    n_exp: int
    topk: int
    first_dense: int
    norm_topk: bool
    routed_scale: float
    eps: float
    vocab: int
    rope_theta: float
    yarn_factor: float
    yarn_orig_max: int
    beta_fast: float
    beta_slow: float
    scaling: float
    cs_mscale: float

    def __init__(self, c: dict[str, Any]) -> None:
        assert c.get("q_lora_rank") is None, "q_lora_rank is not supported"
        assert c["scoring_func"] == "sigmoid" and c["topk_method"] == "noaux_tc", "router"
        assert c["n_group"] == 1 and c.get("topk_group", 1) == 1, "grouped routing"
        assert c["hidden_act"] == "silu" and c["rope_interleave"], "activation / rope layout"
        assert c["is_causal"] is False, "encoder-only checkpoint expected"
        assert c["n_shared_experts"] == 1 and c.get("moe_layer_freq", 1) == 1
        self.hidden = c["hidden_size"]
        self.layers = c["num_hidden_layers"]
        self.heads = c["num_attention_heads"]
        self.nope = c["qk_nope_head_dim"]
        self.rope = c["qk_rope_head_dim"]
        self.vdim = c["v_head_dim"]
        self.qk = self.nope + self.rope
        assert self.vdim == self.qk, "v_head_dim != qk_head_dim needs V padding (not implemented)"
        self.kv_lora = c["kv_lora_rank"]
        self.inter = c["intermediate_size"]
        self.moe_inter = c["moe_intermediate_size"]
        self.n_exp = c["n_routed_experts"]
        self.topk = c["num_experts_per_tok"]
        self.first_dense = c["first_k_dense_replace"]
        self.norm_topk = c["norm_topk_prob"]
        self.routed_scale = float(c["routed_scaling_factor"])
        self.eps = c["rms_norm_eps"]
        self.vocab = c["vocab_size"]
        rs = c["rope_scaling"]
        assert rs.get("rope_type", rs.get("type")) == "yarn"
        self.rope_theta = float(c["rope_theta"])
        self.yarn_factor = float(rs["factor"])
        self.yarn_orig_max = int(rs["original_max_position_embeddings"])
        self.beta_fast = float(rs.get("beta_fast", 32))
        self.beta_slow = float(rs.get("beta_slow", 1))
        m_all = yarn_mscale(self.yarn_factor, float(rs.get("mscale_all_dim", 0)))
        # DeepseekV2Attention: scaling = qk_head_dim**-0.5 * mscale(mscale_all_dim)**2
        self.scaling = self.qk ** -0.5 * m_all * m_all
        # DeepseekScalingRotaryEmbedding: cos/sin times mscale(mscale) / mscale(mscale_all_dim)
        self.cs_mscale = yarn_mscale(self.yarn_factor, float(rs.get("mscale", 1))) / m_all


def load_cfg(path: str) -> Cfg:
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        return Cfg(json.load(f))


def yarn_cos_sin(cfg: Cfg, max_pos: int, device: torch.device) -> Tensor:
    """[max_pos, rope] fp32, cos half then sin half: vLLM's DeepseekScalingRotaryEmbedding cache, op for op
    and ON THE DEVICE (vLLM builds it under the CUDA default device; CPU cos/sin differ by an ulp)."""
    dim, base = cfg.rope, cfg.rope_theta

    def corr_dim(rot: float) -> float:
        return dim * math.log(cfg.yarn_orig_max / (rot * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corr_dim(cfg.beta_fast)), 0)
    high: float = min(math.ceil(corr_dim(cfg.beta_slow)), dim - 1)  # int; vLLM nudges it to a float below
    if low == high:
        high += 0.001
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (cfg.yarn_factor * pos_freqs)
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float, device=device) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    inv = inv_inter * (1 - mask) + inv_extra * mask
    freqs = torch.einsum("i,j -> ij", torch.arange(max_pos, dtype=torch.float32, device=device), inv)
    return torch.cat((freqs.cos() * cfg.cs_mscale, freqs.sin() * cfg.cs_mscale), dim=-1)


def rope_gptj_(x: Tensor, cos_sin: Tensor, pos: Tensor) -> None:
    """In-place interleaved (GPT-J) rotation of x [T, h, rope], computed in fp32 (CPU path)."""
    cos, sin = cos_sin[pos.long()].float().chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    xf = x.float()
    xe, xo = xf[..., 0::2], xf[..., 1::2]
    x.copy_(torch.stack((xe * cos - xo * sin, xo * cos + xe * sin), -1).flatten(-2).to(x.dtype))


class Layer:
    """One decoder layer's weights. Dense layers set gu/dn; MoE layers set the rest of the MLP group."""

    moe: bool
    in_norm: Tensor
    post_norm: Tensor
    q: Tensor
    kv_a: Tensor
    kv_ln: Tensor
    kv_b: Tensor
    o: Tensor
    # dense MLP
    gu: Tensor
    dn: Tensor
    # MoE
    w13: Tensor
    w2: Tensor
    gate_w: Tensor
    gate_bias: Tensor
    sh_gu: Tensor
    sh_dn: Tensor
    seen: set[tuple[int, str]]  # loading only


class GigaEmbedModel:
    def __init__(self, cfg: Cfg, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 max_pos: int = 2048) -> None:
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.cuda = device.type == "cuda"
        if self.cuda and not HAVE_CUDA_OPS:
            raise RuntimeError(f"CUDA kernels unavailable: {_CUDA_OPS_ERR!r}")
        self._taps: dict[str, Tensor] | None = None
        self._li = 0
        # vLLM sizes the cache by original_max_position * factor; any length up to max_pos is the same rows.
        self.cos_sin = yarn_cos_sin(cfg, max_pos, device)
        self.embed: Tensor | None = None
        self.norm_w: Tensor | None = None
        self.layers: list[Layer] = []

    # ---- weights ----------------------------------------------------------------------------
    def load(self, path: str) -> float:
        """Loads model-*.safetensors straight to the device. Returns seconds taken."""
        from safetensors import safe_open

        c, dev, dt = self.cfg, self.device, self.dtype
        layers = [Layer() for _ in range(c.layers)]
        for i, l in enumerate(layers):
            l.moe = i >= c.first_dense
            if l.moe:
                l.w13 = torch.empty(c.n_exp, 2 * c.moe_inter, c.hidden, dtype=dt, device=dev)
                l.w2 = torch.empty(c.n_exp, c.hidden, c.moe_inter, dtype=dt, device=dev)
                l.seen = set()
        part: dict[tuple[int, str], Tensor] = {}
        t0 = time.time()
        files = sorted(glob.glob(os.path.join(path, "model-*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no model-*.safetensors in {path}")
        for fn in files:
            with safe_open(fn, "pt", device=str(dev)) as f:
                for name in f.keys():
                    if not name.startswith("model."):
                        continue
                    if name == "model.embed_tokens.weight":
                        self.embed = f.get_tensor(name).to(dt)
                        continue
                    if name == "model.norm.weight":
                        self.norm_w = f.get_tensor(name).to(dt)
                        continue
                    parts = name.split(".")
                    i = int(parts[2])
                    if i >= c.layers:  # MTP (nextn) layer: not part of the encoder, vLLM skips it too
                        continue
                    l = layers[i]
                    rest = ".".join(parts[3:])
                    w = f.get_tensor(name)
                    if rest.startswith("mlp.experts."):
                        e, which = int(parts[5]), parts[6]
                        if which == "gate_proj":
                            l.w13[e, : c.moe_inter] = w
                        elif which == "up_proj":
                            l.w13[e, c.moe_inter:] = w
                        elif which == "down_proj":
                            l.w2[e] = w
                        else:
                            raise KeyError(name)
                        l.seen.add((e, which))
                    elif rest == "mlp.gate.e_score_correction_bias":
                        l.gate_bias = w.float()
                    else:
                        part[(i, rest)] = w.to(dt)
        for i, l in enumerate(layers):
            def g(k: str) -> Tensor:
                return part.pop((i, k))

            l.in_norm = g("input_layernorm.weight")
            l.post_norm = g("post_attention_layernorm.weight")
            l.q = g("self_attn.q_proj.weight")
            l.kv_a = g("self_attn.kv_a_proj_with_mqa.weight")
            l.kv_ln = g("self_attn.kv_a_layernorm.weight")
            l.kv_b = g("self_attn.kv_b_proj.weight")
            l.o = g("self_attn.o_proj.weight")
            if l.moe:
                if len(l.seen) != 3 * c.n_exp:
                    raise ValueError(f"layer {i}: {len(l.seen)} expert tensors, expected {3 * c.n_exp}")
                del l.seen
                l.gate_w = g("mlp.gate.weight")
                if not hasattr(l, "gate_bias"):
                    raise ValueError(f"layer {i}: no e_score_correction_bias")
                # MergedColumnParallelLinear order: [gate; up]
                l.sh_gu = torch.cat([g("mlp.shared_experts.gate_proj.weight"),
                                     g("mlp.shared_experts.up_proj.weight")], 0).contiguous()
                l.sh_dn = g("mlp.shared_experts.down_proj.weight")
            else:
                l.gu = torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")], 0).contiguous()
                l.dn = g("mlp.down_proj.weight")
        if part:
            raise ValueError(f"unused weights: {sorted(part)[:5]}")
        if self.embed is None or self.norm_w is None:
            raise ValueError("embed_tokens / norm weights missing")
        self.layers = layers
        if self.cuda:
            torch.cuda.synchronize()
        return time.time() - t0

    # ---- ops ----------------------------------------------------------------------------------
    def _rms(self, x: Tensor, w: Tensor) -> Tensor:
        if self.cuda:
            out = torch.empty_like(x)
            vops.rms_norm(out, x, w, self.cfg.eps)
            return out
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.eps)).to(x.dtype) * w

    def _add_rms(self, x: Tensor, res: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        """(rms(x + res), x + res); in place on CUDA like vLLM's fused_add_rms_norm."""
        if self.cuda:
            vops.fused_add_rms_norm(x, res, w, self.cfg.eps)
            return x, res
        s = x + res
        return self._rms(s, w), s

    def _silu_mul(self, x: Tensor) -> Tensor:
        d = x.shape[-1] // 2
        if self.cuda:
            out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
            torch.ops._C.silu_and_mul(out, x)
            return out
        return F.silu(x[..., :d]) * x[..., d:]

    def _route(self, h: Tensor, l: Layer) -> tuple[Tensor, Tensor]:
        c = self.cfg
        logits = F.linear(h, l.gate_w)
        self._tap("mlp.gate", logits)
        if self.cuda:
            w, idx = vops.grouped_topk(logits, 1, 1, c.topk, c.norm_topk, c.routed_scale, l.gate_bias, 1, None)
            return w, idx.to(torch.int32)
        scores = logits.float().sigmoid()
        _, idx = (scores + l.gate_bias).topk(c.topk, dim=-1, sorted=False)
        w = scores.gather(1, idx)
        if c.norm_topk:
            w = w / w.sum(-1, keepdim=True)
        return w * c.routed_scale, idx.to(torch.int32)

    def _experts(self, h: Tensor, l: Layer, w: Tensor, idx: Tensor) -> Tensor:
        if self.cuda:
            return fused_experts(h, l.w13, l.w2, w, idx)
        out = torch.zeros_like(h, dtype=torch.float32)
        for e in range(l.w13.shape[0]):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                y = F.linear(self._silu_mul(F.linear(h[tok], l.w13[e])), l.w2[e]).float()
                out.index_add_(0, tok, y * w[tok, slot].unsqueeze(-1))
        return out.to(h.dtype)

    def _attn_core(self, q: Tensor, k: Tensor, v: Tensor, cu: Tensor, max_len: int, out: Tensor | None) -> Tensor:
        if self.cuda:
            o_fa: Tensor = flash_attn_varlen_func(q, k, v, max_seqlen_q=max_len, cu_seqlens_q=cu,
                                                  max_seqlen_k=max_len, cu_seqlens_k=cu,
                                                  softmax_scale=self.cfg.scaling, causal=False, fa_version=3, out=out)
            return o_fa
        o = torch.zeros(q.shape[0], q.shape[1], v.shape[2], dtype=q.dtype)
        cl = cu.tolist()
        for a, b in zip(cl[:-1], cl[1:]):
            if b > a:
                r = F.scaled_dot_product_attention(q[a:b].transpose(0, 1).float(), k[a:b].transpose(0, 1).float(),
                                                   v[a:b].transpose(0, 1).float(), scale=self.cfg.scaling)
                o[a:b] = r.transpose(0, 1).to(q.dtype)
        return o

    # ---- forward ------------------------------------------------------------------------------
    def _tap(self, name: str, t: Tensor) -> None:
        if self._taps is not None and self._li < 2:
            self._taps[f"{self._li}.{name}"] = t.clone()

    def attention(self, h: Tensor, l: Layer, pos: Tensor, cu: Tensor, max_len: int,
                  attn_out: Tensor | None = None) -> Tensor:
        c = self.cfg
        T, H = h.shape[0], c.heads
        # two GEMMs, not one over [q; kv_a]: vLLM's shapes, so cuBLAS picks the same algorithms
        q = F.linear(h, l.q).view(T, H, c.qk)
        lat = F.linear(h, l.kv_a)  # [T, kv_lora + rope]
        self._tap("self_attn.q_proj", q.reshape(T, -1))
        self._tap("self_attn.kv_a_proj_with_mqa", lat)
        kv_a = self._rms(lat[:, : c.kv_lora].contiguous() if not self.cuda else lat[:, : c.kv_lora], l.kv_ln)
        self._tap("self_attn.kv_a_layernorm", kv_a)
        kv = F.linear(kv_a, l.kv_b)
        self._tap("self_attn.kv_b_proj", kv)
        kv = kv.view(T, H, c.nope + c.vdim)
        q_pe = q[..., c.nope:]
        k_pe = lat[:, c.kv_lora:].unsqueeze(1)  # [T, 1, rope] view
        if self.cuda:  # vLLM v0.34's DeepseekScalingRotaryEmbedding.forward_cuda with FlashInfer present
            apply_rope_with_cos_sin_cache_inplace(positions=pos, query=q_pe, key=k_pe, head_size=c.rope,
                                                  cos_sin_cache=self.cos_sin, is_neox=False)
        else:
            rope_gptj_(q_pe, self.cos_sin, pos)
            rope_gptj_(k_pe, self.cos_sin, pos)
        self._tap("self_attn.rotary_emb.q", q_pe)
        self._tap("self_attn.rotary_emb.k", k_pe)
        k = torch.empty_like(q)
        k[..., : c.nope] = kv[..., : c.nope]
        k[..., c.nope:] = k_pe
        v = kv[..., c.nope:]
        self._tap("self_attn.attn_in.q", q.reshape(T, -1))
        self._tap("self_attn.attn_in.k", k.reshape(T, -1))
        self._tap("self_attn.attn_in.v", v.reshape(T, -1))
        o = self._attn_core(q, k, v, cu, max_len, attn_out)
        self._tap("self_attn.attn", o.reshape(T, -1))
        return F.linear(o.reshape(T, H * c.vdim), l.o)

    def mlp(self, h: Tensor, l: Layer) -> Tensor:
        if not l.moe:
            return F.linear(self._silu_mul(F.linear(h, l.gu)), l.dn)
        w, idx = self._route(h, l)
        shared = F.linear(self._silu_mul(F.linear(h, l.sh_gu)), l.sh_dn)
        self._tap("mlp.shared_experts", shared)
        routed = self._experts(h, l, w, idx)
        self._tap("mlp.routed", routed)
        return routed.add_(shared)

    @torch.inference_mode()
    def hidden(self, ids: Tensor, pos: Tensor, cu: Tensor, max_len: int, attn_out: Tensor | None = None,
               taps: dict[str, Tensor] | None = None) -> Tensor:
        """[T] ids -> [T, hidden] final-normed hidden states. `taps` (dict) collects per-layer outputs."""
        embed, norm_w = self.embed, self.norm_w
        if embed is None or norm_w is None:
            raise RuntimeError("weights not loaded")
        self._taps = taps
        h = F.embedding(ids, embed)
        res = h
        for i, l in enumerate(self.layers):
            self._li = i
            if i == 0:
                h = self._rms(h, l.in_norm)
            else:
                h, res = self._add_rms(h, res, l.in_norm)
            self._tap("input_layernorm", h)
            h = self.attention(h, l, pos, cu, max_len, attn_out)
            if taps is not None:
                taps[f"{i}.attn"] = h.clone()
            h, res = self._add_rms(h, res, l.post_norm)
            h = self.mlp(h, l)
            if taps is not None:
                taps[f"{i}.mlp"] = h.clone()
        h, _ = self._add_rms(h, res, norm_w)
        self._taps = None
        return h

    @torch.inference_mode()
    def pool(self, h: Tensor, seg: Tensor, lens_f: Tensor, nrows: int) -> tuple[Tensor, Tensor]:
        """MEAN pool [T, hidden] by segment id into [nrows, hidden] fp32 -> (raw mean, L2-normalised).
        Rows past len(lens_f) are scratch (padding tokens are pooled into the last row)."""
        acc = torch.zeros(nrows, h.shape[1], dtype=torch.float32, device=h.device)
        acc.index_add_(0, seg, h.float())
        n = lens_f.shape[0]
        acc[:n] /= lens_f.unsqueeze(1)
        return acc, F.normalize(acc, dim=-1)
