"""Embedding throughput + vector dump (aiohttp).
usage: bench.py URL MODEL CONC BATCH NREQ DUMP.npy [WORDS=40] [ENC=bytes]
Texts are deterministic (seed 7) so every engine embeds the same inputs.
ENC=bytes is vLLM's binary encoding (float32 little-endian body + `metadata` header).
"""
import asyncio
import base64
import json
import random
import sys
import time

import aiohttp
import numpy as np

URL, MODEL, DUMP = sys.argv[1], sys.argv[2], sys.argv[6]
CONC, BATCH, NREQ = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
WORDS = int(sys.argv[7]) if len(sys.argv) > 7 else 40
ENC = sys.argv[8] if len(sys.argv) > 8 else "bytes"
rnd = random.Random(7)
VOCAB = ("погода город река книга музыка школа поезд дорога завтра работа новости история "
         "weather city river book music school train road tomorrow work news history "
         "science market energy report garden coffee 2024 2025 million kilometres").split()
texts = [" ".join(rnd.choice(VOCAB) for _ in range(rnd.randint(WORDS // 2, WORDS * 3 // 2)))
         for _ in range(NREQ * BATCH)]


async def one(s, i):
    part = texts[i * BATCH:(i + 1) * BATCH]
    body = {"model": MODEL, "input": part, "encoding_format": ENC}
    if ENC == "bytes":
        body.update(embed_dtype="float32", endianness="little")
    t0 = time.perf_counter()
    async with s.post(URL + "/v1/embeddings", json=body) as r:
        raw = await r.read()
        meta = r.headers.get("metadata")
        status = r.status
    dt = time.perf_counter() - t0
    if status != 200:
        return None, dt, raw[:200].decode("utf-8", "replace"), 0
    if ENC == "bytes":
        m = json.loads(meta)
        vecs = np.frombuffer(raw, "<f4").reshape(len(m["data"]), -1)
        ntok = m["usage"]["prompt_tokens"]
    else:
        j = json.loads(raw)
        ntok = j["usage"]["prompt_tokens"]
        if ENC == "base64":
            vecs = np.stack([np.frombuffer(base64.b64decode(d["embedding"]), "<f4") for d in j["data"]])
        else:
            vecs = np.array([d["embedding"] for d in j["data"]], dtype=np.float32)
    if len(vecs) != len(part):
        return None, dt, f"bad cardinality {len(vecs)} != {len(part)}", 0
    return vecs, dt, None, ntok


async def main():
    sem = asyncio.Semaphore(CONC)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
                                     connector=aiohttp.TCPConnector(limit=CONC)) as s:
        async def run(i):
            async with sem:
                return await one(s, i)

        await asyncio.gather(*(run(i) for i in range(min(8, NREQ))))  # warm-up
        t0 = time.perf_counter()
        res = await asyncio.gather(*(run(i) for i in range(NREQ)))
        wall = time.perf_counter() - t0
    ok = [r for r in res if r[0] is not None]
    errs = [r[2] for r in res if r[0] is None][:3]
    tok = sum(r[3] for r in ok)
    lat = sorted(r[1] for r in ok)
    vecs = np.concatenate([r[0] for r in res[: max(1, 256 // BATCH)] if r[0] is not None])[:256]
    np.save(DUMP, vecs)
    print(json.dumps({"wall_s": round(wall, 1), "conc": CONC, "batch": BATCH, "enc": ENC, "ok": len(ok), "n": NREQ,
                      "inputs_s": round(len(ok) * BATCH / wall, 1), "tok_s": round(tok / wall),
                      "p50_s": round(lat[len(lat) // 2], 4) if lat else None,
                      "p95_s": round(lat[int(len(lat) * 0.95) - 1], 4) if lat else None, "errors": errs}), flush=True)


asyncio.run(main())
