#!/usr/bin/env python3
"""Tune vLLM 0.34's Triton fused_moe tile config for ONE shape: giga-serve's MoE layer (E=64 routed experts,
hidden 1536, moe_intermediate 1280, top-k from the checkpoint config).

  python3 tools/tune_moe.py MODEL_DIR OUT.json [--src REPO] [--layer N] [--quick] [--ms 16,32,...|server]
                      [--max-tokens 16384] [--graph-max-tokens 4096] [--jobs J] [--budget-min MIN]
                      [--hidden fwd|rand] [--seqlen 128] [--device 0] [--seed 0]

MODEL_DIR checkpoint dir (config.json + model-*.safetensors)
OUT.json  tuned config, vLLM file format (keys = M as strings), fastest overall per M.
          Also written: OUT.bitexact.json (fastest config per M whose output is torch.equal to vLLM's shipped
          config's) and OUT.candidates.json (every measurement). To install the bit-identical result:
          cp OUT.bitexact.json "moe_configs/E=64,N=1280,device_name=<GPU>.json" (the name is printed at the end).

How a config is forced (no monkeypatching): vllm.model_executor.layers.fused_moe.override_config(cfg), a context
manager (fused_moe/__init__.py:53-63). fused_experts -> fused_experts_impl -> try_get_optimal_moe_config ->
_get_configured_moe_config returns it before any tuned-file lookup (fused_moe/fused_moe.py:1526-1530, 1564-1568,
1854-1863). The dict is used for BOTH GEMMs and for moe_align_block_size's BLOCK_SIZE_M (fused_moe.py:1904-1971);
BLOCK_SIZE_K is popped and SPLIT_K forced to 1 (fused_moe.py:878-880), the rest goes to the Triton launch as
**config (BLOCK_SIZE_M/N, GROUP_SIZE_M, num_warps, num_stages).
vLLM's override_config has no try/finally, so `forced_config` below wraps it: a failing (OutOfResources etc.)
candidate must not leave the override set.

Method
  1. Load layers 0..N of the real checkpoint with giga_model.GigaEmbedModel.load (cfg.layers is cut to N+1, the loader
     skips the rest like it skips the MTP layer: ~10 GB, not 20), run the real layers 0..N-1 + layer N's attention on
     random token ids to get realistic MoE-input hidden states, route with GigaEmbedModel._route (real gate weights,
     vops.grouped_topk), then free everything except layer N.
  2. Per M: the default = whatever fused_experts picks today (no override). Its output is the bit-exactness reference.
  3. Stage A: every tile (BM,BN,BK,warps,stages) of the search space (pruned by shared-memory / register / tokens-per-
     expert heuristics) with one GROUP_SIZE_M. Stage B: the best few tiles per M (overall and bit-identical) x GROUP_SIZE_M.
     Finalists (top few of each ranking + default) are re-timed interleaved, 3 rounds, min taken. A candidate replaces
     the default only if >= 1% faster.
  4. Timing = CUDA-graph replay of the full fused_experts call (align + GEMM1 + silu_and_mul + GEMM2 + moe_sum), mean of
     n replays, min over repeats. Triton compiles are done up front by a pool of worker processes sharing the Triton
     disk cache (--jobs), the parent then only loads cache hits.
"""
import argparse
import contextlib
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import torch

KEYS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages")
DEFAULT_MS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# What the server actually calls fused_experts with: giga_engine.T_BUCKETS (padded graph steps) + eager step sizes.
SERVER_MS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
SMEM_BUDGET = 200 * 1024  # H200 allows 227 KB dynamic smem per block; keep margin for layout-conversion scratch
ACC_REGS_MAX = 160  # fp32 accumulator registers per thread (BM*BN / (32*warps)); above that it only spills
MIN_GAIN = 0.01  # a candidate must beat the default by 1% to replace it (timing noise on a shared GPU)
GM_QUICK = (1, 16)
GM_FULL = (1, 4, 8, 16, 32, 64)


def key_of(c: dict) -> tuple:
    return tuple(int(c[k]) for k in KEYS)


def cfg_of(k: tuple) -> dict:
    return dict(zip(KEYS, k))


def fmt(k) -> str:
    if k is None:
        return "-"
    bm, bn, bk, gm, w, s = k
    return f"BM{bm} BN{bn} BK{bk} GM{gm} w{w} s{s}"


@contextlib.contextmanager
def forced_config(cfg: dict):
    """vLLM's own override_config (fused_moe/__init__.py:53-63) with the try/finally it lacks."""
    from vllm.model_executor.layers.fused_moe import override_config

    cm = override_config(cfg)
    cm.__enter__()
    try:
        yield
    finally:
        cm.__exit__(None, None, None)


def short_err(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}".replace("\n", " ")
    return s[:200]


def fatal_cuda(msg: str) -> bool:
    m = msg.lower()
    return "illegal memory" in m or "misaligned" in m or "device-side assert" in m or "unspecified launch" in m


# ----------------------------------------------------------------------------------------------
# search space
# ----------------------------------------------------------------------------------------------
def tile_space(quick: bool) -> list[tuple]:
    if quick:
        bms, bns, bks, ws, ss = (16, 32, 64, 128), (64, 128), (64, 128), (4, 8), (3, 4)
    else:  # vLLM benchmarks/kernels/benchmark_moe.py conventions (BM 16..256, BN 32..256, BK 64..256, w 4/8, s 2..5)
        bms, bns, bks, ws, ss = (16, 32, 64, 128, 256), (32, 64, 128, 256), (64, 128, 256), (4, 8), (2, 3, 4, 5)
    out = []
    for bm, bn, bk, w, s in itertools.product(bms, bns, bks, ws, ss):
        if s * (bm + bn) * bk * 2 > SMEM_BUDGET:
            continue
        if bm * bn / (32 * w) > ACC_REGS_MAX:
            continue
        if w == 8 and bm * bn < 8192:
            continue
        out.append((bm, bn, bk, w, s))
    return out


def next_pow2(x: float) -> int:
    return 1 << max(0, math.ceil(x) - 1).bit_length() if x > 1 else 1



def bm_allowed(bm: int, M: int, topk: int, E: int) -> bool:
    """A BLOCK_SIZE_M far above the average tokens per expert only pads; skip those."""
    avg = M * topk / E
    return bm <= max(16, 4 * next_pow2(avg))


def gm_stage_a(M: int, topk: int, E: int) -> int:
    return 1 if M * topk / E < 32 else 8


# ----------------------------------------------------------------------------------------------
# worker pool: Triton compile in parallel (shared on-disk Triton cache)
# ----------------------------------------------------------------------------------------------
_W: dict = {}


def _winit(device: int, hidden: int, inter: int, topk: int) -> None:
    try:
        torch.cuda.set_device(device)
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts  # noqa: F401

        # E_w must keep `EM = M*topk + E*(BM-1)` divisible by 16 exactly when the real E=64 does (Triton specialises
        # integer args on divisibility by 16): any multiple of 16 works. Expert count is not a kernel argument.
        e_w = max(16, ((2 * topk + 15) // 16) * 16)
        dev = torch.device("cuda", device)
        g = torch.Generator(device=dev).manual_seed(1)
        m = 1024
        _W["w1"] = (torch.randn(e_w, 2 * inter, hidden, device=dev, dtype=torch.bfloat16, generator=g) * 0.02)
        _W["w2"] = (torch.randn(e_w, hidden, inter, device=dev, dtype=torch.bfloat16, generator=g) * 0.02)
        _W["h"] = torch.randn(m, hidden, device=dev, dtype=torch.bfloat16, generator=g)
        _W["idx"] = torch.rand(m, e_w, device=dev, generator=g).topk(topk, dim=-1).indices.to(torch.int32)
        _W["w"] = torch.softmax(torch.rand(m, topk, device=dev, generator=g), dim=-1)
        _W["fe"] = fused_experts
    except BaseException as e:  # reported per task; the parent falls back to compiling itself
        _W["err"] = short_err(e)


def _wwarm(keys: list) -> list:
    res = []
    for k in keys:
        if "err" in _W:
            res.append((k, False, "worker init: " + _W["err"]))
            continue
        try:
            with forced_config(cfg_of(k)):
                out = _W["fe"](_W["h"], _W["w1"], _W["w2"], _W["w"], _W["idx"])
            torch.cuda.synchronize()
            res.append((k, True, "" if bool(torch.isfinite(out).all()) else "non-finite"))
        except Exception as e:
            res.append((k, False, short_err(e)))
    return res


class Compiler:
    """Pre-compiles configs in worker processes; compile_all returns (ok_keys, invalid{key: msg}, timed_out_keys)."""

    def __init__(self, jobs: int, device: int, hidden: int, inter: int, topk: int):
        self.args = (device, hidden, inter, topk)
        self.jobs = jobs
        self.pool = None
        self.broken = False
        if jobs > 1:
            self._start()

    def _start(self):
        self.pool = mp.get_context("spawn").Pool(self.jobs, initializer=_winit, initargs=self.args)

    def compile_all(self, keys: list, deadline: float, label: str):
        if self.jobs <= 1 or self.broken or not keys:
            return None  # parent compiles lazily
        if self.pool is None:
            self._start()
        chunks = [keys[i:i + 4] for i in range(0, len(keys), 4)]
        it = self.pool.imap_unordered(_wwarm, chunks)
        ok, invalid, done = set(), {}, 0
        t0 = time.time()
        while done < len(chunks):
            left = deadline - time.time()
            try:
                res = it.next(timeout=max(left, 0.1))
            except mp.TimeoutError:
                print(f"[{label}] compile phase hit its deadline: {done}/{len(chunks)} chunks done", flush=True)
                self.pool.terminate()
                self.pool = None
                break
            done += 1
            for k, good, msg in res:
                if good:
                    ok.add(k)
                elif msg.startswith("worker init"):
                    self.broken = True
                    invalid[k] = msg
                else:
                    invalid[k] = msg
            if done % 20 == 0:
                print(f"[{label}] compiled {done * 4}/{len(keys)} configs  {time.time() - t0:.0f}s", flush=True)
        if self.broken:
            print("[compile] worker init failed (" + next(iter(invalid.values())) + "); compiling serially", flush=True)
            return None
        todo_left = [k for k in keys if k not in ok and k not in invalid]
        return ok, {k: v for k, v in invalid.items()}, todo_left

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None


# ----------------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------------
_POOL = None


def time_fn(fn, target_ms: float = 10.0, reps: int = 2) -> float:
    """us per call: CUDA-graph replay (falls back to eager launches), mean of n back-to-back calls, min of reps."""
    global _POOL
    torch.cuda.synchronize()
    launch, g = fn, None
    try:
        if _POOL is None:
            _POOL = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=_POOL):
            fn()
        launch = g.replay
    except Exception:
        g, launch = None, fn
        torch.cuda.synchronize()
    for _ in range(2):
        launch()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    launch()
    e1.record()
    torch.cuda.synchronize()
    est = max(e0.elapsed_time(e1), 1e-3)
    n = int(max(3, min(200, target_ms / est)))
    best = float("inf")
    for _ in range(reps):
        e0.record()
        for _ in range(n):
            launch()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / n)
    del g
    return best * 1000.0


# ----------------------------------------------------------------------------------------------
# model / data
# ----------------------------------------------------------------------------------------------
def moe_inputs(G, model, L: int, tmax: int, seqlen: int, mode: str, seed: int):
    """[tmax, hidden] bf16 = input of layer L's MoE block (after post_attention_layernorm), contiguous."""
    import torch.nn.functional as F

    dev, c = model.device, model.cfg
    layer = model.layers[L]
    g = torch.Generator(device=dev).manual_seed(seed)
    if mode == "rand":
        h = torch.randn(tmax, c.hidden, device=dev, dtype=torch.bfloat16, generator=g)
        return model._rms(h, layer.post_norm).contiguous()
    ids = torch.randint(1000, c.vocab - 1000, (tmax,), device=dev, generator=g, dtype=torch.int64)
    pos = (torch.arange(tmax, device=dev) % seqlen).to(torch.int64)
    lens = [seqlen] * (tmax // seqlen) + ([tmax % seqlen] if tmax % seqlen else [])
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    cu = cu.to(dev)
    # mirrors GigaEmbedModel.hidden() up to the MLP of layer L
    h = F.embedding(ids, model.embed)
    res = h
    for i in range(L + 1):
        l = model.layers[i]
        model._li = i
        if i == 0:
            h = model._rms(h, l.in_norm)
        else:
            h, res = model._add_rms(h, res, l.in_norm)
        h = model.attention(h, l, pos, cu, seqlen)
        h, res = model._add_rms(h, res, l.post_norm)
        if i == L:
            return h.contiguous()
        h = model.mlp(h, l)
    raise AssertionError("unreachable")


class MData:
    def __init__(self, M, h, w, idx, ref, default_cfg, default_key):
        self.M, self.h, self.w, self.idx, self.ref = M, h, w, idx, ref
        self.default_cfg, self.default_key = default_cfg, default_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="directory holding giga_model.py (default: the repo root above tools/)")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ms", default=None, help="comma list of M, or 'server' (the step sizes giga_engine runs with "
                                               "--max-tokens / --graph-max-tokens)")
    ap.add_argument("--max-tokens", type=int, default=16384, help="server --max-num-batched-tokens (for --ms server)")
    ap.add_argument("--graph-max-tokens", type=int, default=4096, help="server --graph-max-tokens (for --ms server)")
    ap.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)),
                    help="compile worker processes (1 = compile serially in this process)")
    ap.add_argument("--budget-min", type=float, default=None, help="wall budget for the search (default 18 / quick 2.5)")
    ap.add_argument("--hidden", choices=("fwd", "rand"), default="fwd")
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # The reference ("default") must be vLLM's shipped config, not a previously tuned file: point vLLM's
    # tuned-config lookup at an empty folder before giga_model (which sets it with setdefault) is imported.
    import tempfile
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = tempfile.mkdtemp(prefix="tune_moe_empty_")
    sys.path.insert(0, a.src)
    if a.ms == "server":
        from giga_engine import T_BUCKETS
        g = a.graph_max_tokens
        graph = [t for t in T_BUCKETS if t < g] + [g] if g else []
        eager = [t for t in T_BUCKETS + (12288, 16384, 24576, 32768, a.max_tokens) if g < t <= a.max_tokens]
        ms = graph + eager
    else:
        ms = [int(x) for x in a.ms.split(",")] if a.ms else DEFAULT_MS
    ms = sorted(set(ms))
    budget = (a.budget_min if a.budget_min is not None else (2.5 if a.quick else 18.0)) * 60.0

    import giga_model as G
    from vllm.model_executor.layers.fused_moe.fused_moe import (fused_experts, get_config_file_name,
                                                                get_moe_configs, try_get_optimal_moe_config)
    import triton

    torch.cuda.set_device(a.device)
    dev = torch.device("cuda", a.device)
    cfg = G.load_cfg(a.model)
    E, topk, hidden, inter = cfg.n_exp, cfg.topk, cfg.hidden, cfg.moe_inter
    print(f"shape: E={E} topk={topk} hidden={hidden} moe_inter={inter}  Ms={ms}  quick={a.quick} "
          f"jobs={a.jobs} budget={budget / 60:.1f}min", flush=True)

    comp = Compiler(a.jobs, a.device, hidden, inter, topk)  # workers import vllm while we load weights

    # ---- real weights of layer N (+ the layers below it, only to make realistic hidden states) ----
    if a.layer < cfg.first_dense or a.layer >= cfg.layers:
        raise SystemExit(f"--layer must be a MoE layer in [{cfg.first_dense}, {cfg.layers})")
    cfg.layers = a.layer + 1  # GigaEmbedModel.load skips layers >= cfg.layers (like the MTP layer)
    model = G.GigaEmbedModel(cfg, dev)
    t0 = time.time()
    print(f"load layers 0..{a.layer}: {model.load(a.model):.1f}s", flush=True)
    tmax = max(ms)
    with torch.inference_mode():
        try:
            hid = moe_inputs(G, model, a.layer, tmax, a.seqlen, a.hidden, a.seed)
        except Exception as e:
            print(f"hidden-state forward failed ({short_err(e)}); falling back to --hidden rand", flush=True)
            hid = moe_inputs(G, model, a.layer, tmax, a.seqlen, "rand", a.seed)
    layer = model.layers[a.layer]
    model.layers, model.embed, model.norm_w = [], None, None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    w13, w2 = layer.w13, layer.w2
    assert w13.shape == (E, 2 * inter, hidden) and w2.shape == (E, hidden, inter), (w13.shape, w2.shape)
    print(f"weights ready ({time.time() - t0:.0f}s); gpu mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    # ---- per-M data: routing, default output (reference), default config ----
    data: dict[int, MData] = {}
    stats = {}
    with torch.inference_mode():
        for M in ms:
            h = hid[:M].contiguous()
            w, idx = model._route(h, layer)
            ref = fused_experts(h, w13, w2, w, idx)
            ref2 = fused_experts(h, w13, w2, w, idx)
            torch.cuda.synchronize()
            if not torch.equal(ref, ref2):
                print(f"WARNING: default config is not deterministic at M={M}; bit-identity flags are unreliable",
                      flush=True)
            d = try_get_optimal_moe_config(w13.size(), w2.size(), topk, None, M)
            dk = key_of(d) if all(k in d for k in KEYS) else None
            cnt = torch.bincount(idx.view(-1).long(), minlength=E).float()
            stats[M] = {"max_over_mean_tokens_per_expert": float(cnt.max() / cnt.mean()),
                        "empty_experts": int((cnt == 0).sum())}
            data[M] = MData(M, h, w, idx, ref, d, dk)
    del hid
    print("routing balance (max/mean tokens per expert) at largest M: "
          f"{stats[tmax]['max_over_mean_tokens_per_expert']:.2f}, empty experts {stats[tmax]['empty_experts']}",
          flush=True)

    # ---- search ----
    clock0 = time.time()
    deadline_a = clock0 + 0.50 * budget
    deadline_b = clock0 + 0.78 * budget
    rec: dict[int, dict[tuple, dict]] = {M: {} for M in ms}
    invalid: dict[tuple, str] = {}
    rng = random.Random(a.seed)

    def measure(k: tuple, M: int, target_ms: float, reps: int):
        d = data[M]
        c = cfg_of(k)

        def fn():
            with forced_config(c):
                return fused_experts(d.h, w13, w2, d.w, d.idx)

        try:
            out = fn()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError("non-finite output")
            equal = torch.equal(out, d.ref)
            diff = (out.float() - d.ref.float())
            maxdiff = float(diff.abs().max())
            frac = float((out != d.ref).float().mean())
            us = time_fn(fn, target_ms, reps)
        except Exception as e:
            msg = short_err(e)
            if fatal_cuda(msg):
                raise
            invalid[k] = msg
            return None
        return {"us": us, "equal": equal, "maxdiff": maxdiff, "frac_diff": frac}

    def evaluate(keys_by_m: dict[tuple, list], deadline: float, label: str, compiled):
        keys = sorted(keys_by_m)
        rng.shuffle(keys)
        n_done, t0_ = 0, time.time()
        with torch.inference_mode():
            for k in keys:
                if time.time() > deadline:
                    print(f"[{label}] search deadline reached after {n_done}/{len(keys)} configs", flush=True)
                    break
                if k in invalid or (compiled is not None and k not in compiled):
                    continue
                for M in sorted(keys_by_m[k]):
                    if k in rec[M] or k in invalid:
                        continue
                    r = measure(k, M, 8.0, 2)
                    if r is not None:
                        rec[M][k] = r
                n_done += 1
                if n_done % 40 == 0:
                    print(f"[{label}] {n_done}/{len(keys)} configs benchmarked  {time.time() - t0_:.0f}s", flush=True)

    def run_phase(keys_by_m: dict[tuple, list], deadline: float, label: str):
        keys = sorted(keys_by_m)
        rng.shuffle(keys)
        t_start = time.time()
        res = comp.compile_all(keys, t_start + 0.6 * (deadline - t_start), label)  # rest of the window = timing
        compiled = None
        if res is not None:
            ok, bad, left = res
            invalid.update(bad)
            compiled = ok
            print(f"[{label}] compiled ok={len(ok)} invalid={len(bad)} not-reached={len(left)}", flush=True)
        evaluate({k: keys_by_m[k] for k in keys}, deadline, label, compiled)

    gm_list = GM_QUICK if a.quick else GM_FULL
    tiles = tile_space(a.quick)
    seeds: set[tuple] = set()
    try:  # every entry of the shipped tuned file is a known-good candidate for every M
        shipped = get_moe_configs(E, inter, None, None, None)
        for v in (shipped or {}).values():
            if all(k in v for k in KEYS):
                seeds.add(key_of(v))
    except Exception:
        pass

    # stage A
    ka: dict[tuple, list] = {}
    for M in ms:
        gm = gm_stage_a(M, topk, E)
        for bm, bn, bk, w, s in tiles:
            if bm_allowed(bm, M, topk, E):
                ka.setdefault((bm, bn, bk, gm, w, s), []).append(M)
        for sk in seeds:
            if bm_allowed(sk[0], M, topk, E):
                ka.setdefault(sk, []).append(M)
    ka = {k: sorted(set(v)) for k, v in ka.items()}
    print(f"stage A: {len(ka)} distinct configs, {sum(len(v) for v in ka.values())} (config, M) runs", flush=True)
    run_phase(ka, deadline_a, "A")

    # stage B: GROUP_SIZE_M around the best tiles
    nb = 2 if a.quick else 5
    kb: dict[tuple, list] = {}
    for M in ms:
        valid = sorted(rec[M].items(), key=lambda kv: kv[1]["us"])
        top = [k for k, _ in valid[:nb]]
        top += [k for k, r in valid if r["equal"]][:nb]
        for k in set(top):
            for gm in gm_list:
                k2 = (k[0], k[1], k[2], gm, k[4], k[5])
                if k2 not in rec[M]:
                    kb.setdefault(k2, []).append(M)
    print(f"stage B: {len(kb)} distinct configs, {sum(len(v) for v in kb.values())} (config, M) runs", flush=True)
    run_phase(kb, deadline_b, "B")
    comp.close()

    # ---- finalists: interleaved re-measure against the default ----
    nfin = 2 if a.quick else 3
    results = {}
    with torch.inference_mode():
        for M in ms:
            d = data[M]
            valid = sorted(rec[M].items(), key=lambda kv: kv[1]["us"])
            cands: list = []
            for k, _ in valid[:nfin]:
                cands.append(k)
            for k, r in valid:
                if r["equal"] and k not in cands and sum(1 for c in cands if rec[M][c]["equal"]) < nfin:
                    cands.append(k)
            cands = [k for k in cands if k != d.default_key]

            def mk(k):
                c = cfg_of(k)

                def fn():
                    with forced_config(c):
                        return fused_experts(d.h, w13, w2, d.w, d.idx)

                return fn

            fns = {None: (lambda: fused_experts(d.h, w13, w2, d.w, d.idx))}
            fns.update({k: mk(k) for k in cands})
            best = {k: float("inf") for k in fns}
            for _ in range(3):
                for k, fn in fns.items():
                    best[k] = min(best[k], time_fn(fn, 25.0, 2))
            d_us = best[None]
            row = {"default_us": d_us, "default_cfg": d.default_cfg, "default_key": d.default_key,
                   "n_valid": len(valid), "n_bitexact": sum(1 for _, r in valid if r["equal"])}

            def pick(pred):
                pool_ = [(best[k], k) for k in cands if pred(rec[M][k])]
                if not pool_:
                    return None
                us, k = min(pool_)
                return (us, k) if us < d_us * (1 - MIN_GAIN) else None

            for name, pred in (("any", lambda r: True), ("exact", lambda r: r["equal"])):
                p = pick(pred)
                if p is None:
                    row[name] = {"us": d_us, "key": d.default_key, "default": True, "equal": True,
                                 "maxdiff": 0.0, "frac_diff": 0.0}
                else:
                    r = rec[M][p[1]]
                    row[name] = {"us": p[0], "key": p[1], "default": False, "equal": r["equal"],
                                 "maxdiff": r["maxdiff"], "frac_diff": r["frac_diff"]}
            results[M] = row
            print(f"[final] M={M:6d} default {d_us:9.1f}us  any {row['any']['us']:9.1f}us  "
                  f"exact {row['exact']['us']:9.1f}us", flush=True)

    # ---- outputs ----
    tv = triton.__version__

    def cfg_json(M: int, which: str) -> dict:
        r = results[M][which]
        c = results[M]["default_cfg"] if r["default"] else cfg_of(r["key"])
        return {k: int(c[k]) for k in KEYS if k in c}

    stem = a.out[:-5] if a.out.endswith(".json") else a.out
    paths = {"any": a.out, "exact": stem + ".bitexact.json"}
    for which, path in paths.items():
        doc = {str(M): cfg_json(M, which) for M in ms}
        doc["triton_version"] = tv
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=4)
    cand_path = stem + ".candidates.json"
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump({
            "shape": {"E": E, "topk": topk, "hidden": hidden, "moe_inter": inter, "layer": a.layer},
            "triton": tv, "routing": stats, "invalid": {fmt(k): v for k, v in invalid.items()},
            "per_m": {str(M): {
                "default_us": results[M]["default_us"],
                "default_cfg": {k: results[M]["default_cfg"].get(k) for k in KEYS},
                "any": results[M]["any"] | {"key": fmt(results[M]["any"]["key"])},
                "exact": results[M]["exact"] | {"key": fmt(results[M]["exact"]["key"])},
                "candidates": [{"cfg": fmt(k), "us": r["us"], "speedup": results[M]["default_us"] / r["us"],
                                "bit_identical": r["equal"], "max_abs_diff": r["maxdiff"],
                                "frac_elems_differing": r["frac_diff"]}
                               for k, r in sorted(rec[M].items(), key=lambda kv: kv[1]["us"])],
            } for M in ms}}, f, indent=1, default=str)

    # ---- summary ----
    print()
    print(f"{'M':>6} {'default us':>10} | {'best any us':>11} {'x':>5} {'bit=':>4} {'maxdiff':>8} {'config':<34} | "
          f"{'best exact us':>13} {'x':>5} {'config':<34} | exact/valid")
    tot = {"d": 0.0, "a": 0.0, "e": 0.0}
    for M in ms:
        r = results[M]
        aa, ee = r["any"], r["exact"]
        print(f"{M:>6} {r['default_us']:>10.1f} | {aa['us']:>11.1f} {r['default_us'] / aa['us']:>5.2f} "
              f"{'yes' if aa['equal'] else 'NO':>4} {aa['maxdiff']:>8.3g} {fmt(aa['key']) + (' (default)' if aa['default'] else ''):<34} | "
              f"{ee['us']:>13.1f} {r['default_us'] / ee['us']:>5.2f} "
              f"{fmt(ee['key']) + (' (default)' if ee['default'] else ''):<34} | {r['n_bitexact']}/{r['n_valid']}")
        tot["d"] += r["default_us"]
        tot["a"] += aa["us"]
        tot["e"] += ee["us"]
    print(f"sum over Ms: default {tot['d']:.0f}us  any {tot['a']:.0f}us ({tot['d'] / tot['a']:.3f}x)  "
          f"exact {tot['e']:.0f}us ({tot['d'] / tot['e']:.3f}x)")
    print(f"invalid/failed configs: {len(invalid)}   total {time.time() - clock0:.0f}s search")
    print(f"\nwrote {paths['any']} (fastest), {paths['exact']} (fastest bit-identical), {cand_path}")
    print("install as: <VLLM_TUNED_CONFIG_FOLDER>/" + get_config_file_name(E, inter, None))


if __name__ == "__main__":
    main()
