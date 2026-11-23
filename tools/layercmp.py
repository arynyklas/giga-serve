"""Per-layer numerics: vLLM's own DeepSeek-V2 model (in-process, eager, serving args) vs giga_model.

  python layercmp.py MODEL_DIR [N_TEXTS]

Runs one vLLM step over N token-id prompts with forward hooks on every layer (and every
sub-module of layers 0-1), runs giga_model on the same flat batch in vLLM's order, and prints the
max abs difference per tap and the final cosine. Expected on the image giga-serve pins: every tap
bit-equal; vectors bit-equal up to the order of fp32 atomics in the mean pool.
"""
import os
import random
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
import torch.nn.functional as F

MODEL_DIR = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..")]


def texts(n, words=40):
    rnd = random.Random(7)
    vocab = ("погода город река книга музыка школа поезд дорога завтра работа новости история "
             "weather city river book music school train road tomorrow work news history "
             "science market energy report garden coffee 2024 2025 million kilometres").split()
    return [" ".join(rnd.choice(vocab) for _ in range(rnd.randint(words // 2, words * 3 // 2))) for _ in range(n)]


def main():
    from tokenizers import Tokenizer
    from vllm import LLM
    from vllm.config import PoolerConfig

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    seqs = [e.ids for e in tok.encode_batch(texts(N))]
    seqs[0] = seqs[0][:1]  # a BOS-only (empty text) sequence
    llm = LLM(model=MODEL_DIR, runner="pooling", convert="embed", trust_remote_code=True,
              hf_overrides={"model_type": "deepseek_v3", "auto_map": None},
              pooler_config=PoolerConfig(pooling_type="MEAN", use_activation=True),
              max_model_len=2048, max_num_batched_tokens=16384, max_num_seqs=512, enforce_eager=True,
              gpu_memory_utilization=0.3, enable_prefix_caching=False)
    taps, ids_seen = {}, []

    def put(key, o):
        if isinstance(o, tuple):
            if key.endswith("rotary_emb"):
                taps[key + ".q"], taps[key + ".k"] = o[0].detach().clone(), o[1].detach().clone()
                return
            o = o[0]
        if o is not None:
            taps[key] = o.detach().clone()

    def install(model):
        hs = []
        for name, mod in model.named_modules():
            parts = name.split(".")
            if "layers" not in parts:
                if name.endswith("embed_tokens"):
                    hs.append(mod.register_forward_pre_hook(lambda m, a: ids_seen.append(a[0].detach().clone())))
                continue
            li = parts.index("layers") + 1
            if li >= len(parts) or not parts[li].isdigit():
                continue
            sub = ".".join(parts[li + 1:])
            if sub in ("self_attn", "mlp"):
                key = f"{parts[li]}.{'attn' if sub == 'self_attn' else 'mlp'}"
                hs.append(mod.register_forward_hook(lambda m, a, o, key=key: put(key, o)))
            elif int(parts[li]) < 2 and sub:
                key = f"{parts[li]}.{sub}"
                hs.append(mod.register_forward_hook(lambda m, a, o, key=key: put(key, o)))
                if sub == "self_attn.attn":
                    def pre(m, a, key=key):
                        for n, t in zip("qkv", a[:3]):
                            if t is not None:
                                taps[f"{key}_in.{n}"] = t.detach().clone()
                    hs.append(mod.register_forward_pre_hook(pre))
        return len(hs)

    print("hooks", llm.apply_model(install), flush=True)
    outs = llm.embed([{"prompt_token_ids": s} for s in seqs], use_tqdm=False)
    ref_emb = torch.stack([torch.as_tensor(o.outputs.embedding) for o in outs]).float()
    flat = ids_seen[-1].cpu().tolist()
    assert len(flat) == sum(len(s) for s in seqs), (len(flat), sum(len(s) for s in seqs))
    order, p, left = [], 0, set(range(N))
    while p < len(flat):  # vLLM may reorder requests inside a step: recover its order
        j = next(j for j in sorted(left, key=lambda j: -len(seqs[j])) if flat[p:p + len(seqs[j])] == seqs[j])
        order.append(j)
        left.remove(j)
        p += len(seqs[j])
    print("vLLM step order", order[:16], "...", flush=True)
    ref_emb = ref_emb[order]
    seqs = [seqs[j] for j in order]
    mine_flat = torch.tensor(flat)
    vtaps = {k: v.float().cpu() for k, v in taps.items()}

    import giga_model as G
    dev = torch.device("cuda:0")
    cfg = G.load_cfg(MODEL_DIR)
    m = G.GigaEmbedModel(cfg, dev, max_pos=2048)
    print("load s", round(m.load(MODEL_DIR), 1), flush=True)
    lens = [len(s) for s in seqs]
    ids = mine_flat.to(dev)
    cu = torch.zeros(N + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lens).cumsum(0)
    seg = torch.repeat_interleave(torch.arange(N), torch.tensor(lens)).to(dev)
    pos = (torch.arange(len(ids)) - cu[:-1].long()[seg.cpu()]).to(dev)
    cu = cu.to(dev)
    mt = {}
    h = m.hidden(ids, pos, cu, max(lens), taps=mt)
    _, emb = m.pool(h, seg, torch.tensor(lens, dtype=torch.float32, device=dev), N)
    cos = F.cosine_similarity(emb.cpu(), ref_emb)
    for key in sorted(vtaps, key=lambda k: (int(k.split(".")[0]), k)):
        if int(key.split(".")[0]) > 1:
            continue
        v = vtaps[key]
        mine = mt.get(key)
        if mine is None or mine.numel() != v.numel():
            print(f"   {key:40s} vLLM {tuple(v.shape)} ours {None if mine is None else tuple(mine.shape)}")
            continue
        d = (mine.float().cpu().reshape(v.shape) - v).abs()
        print(f"   {key:40s} {tuple(v.shape)} max|d| {d.max().item():.3g} frac!=0 {(d > 0).float().mean().item():.4f}")
    first, worst = None, 0.0
    for i in range(cfg.layers):
        for kind in ("attn", "mlp"):
            key = f"{i}.{kind}"
            d = (mt[key].float().cpu() - vtaps[key]).abs().max().item()
            worst = max(worst, d)
            if d > 0 and first is None:
                first = (key, d)
    print(f"layer outputs: first diff {first}, max |d| {worst:.3g} | emb cos min {cos.min():.7f} mean "
          f"{cos.mean():.8f}, bit-equal vectors {(emb.cpu() == ref_emb).all(1).sum().item()}/{N}", flush=True)


if __name__ == "__main__":
    main()
