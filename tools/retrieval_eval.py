"""Retrieval A/B for /v1/embeddings engines (numpy + stdlib only).

  retrieval_eval.py embed URL MODEL DATASET OUT.npy [CONC] [BATCH] [TRUNC]
      embeds the dataset's queries (with instruction) and corpus, in a fixed order, base64.
  retrieval_eval.py score DATASET A.npy [B.npy ...]
      nDCG@10 / recall@10 / recall@100 / MRR@10 per file; for each B vs A: per-vector cosine,
      per-query top-10 overlap and top-1 agreement.

Datasets (downloaded to $DS_DIR, default /tmp/ds, on first use): rubq (ai-forever/rubq-retrieval,
ru), scifact (mteb/scifact, en).
"""
import base64
import concurrent.futures as cf
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

DS = os.environ.get("DS_DIR", "/tmp/ds")
HF = "https://huggingface.co/datasets"
SETS = {
    "rubq": dict(
        corpus=f"{HF}/ai-forever/rubq-retrieval/resolve/main/data/corpus.jsonl.zip",
        queries=f"{HF}/ai-forever/rubq-retrieval/resolve/main/data/queries.jsonl",
        qrels=f"{HF}/ai-forever/rubq-retrieval/resolve/main/data/test.jsonl",
        instruct="Given a question, retrieve Wikipedia passages that answer the question",
    ),
    "scifact": dict(
        corpus=f"{HF}/mteb/scifact/resolve/main/corpus.jsonl",
        queries=f"{HF}/mteb/scifact/resolve/main/queries.jsonl",
        qrels=f"{HF}/mteb/scifact/resolve/main/qrels/test.jsonl",
        instruct="Given a scientific claim, retrieve documents that support or refute the claim",
    ),
}


def fetch(url):
    os.makedirs(DS, exist_ok=True)
    path = os.path.join(DS, url.replace("/", "_")[-120:])
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=300) as r, open(path + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(path + ".tmp", path)
    return path


def jsonl(path):
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        data = z.read(z.namelist()[0])
    else:
        data = open(path, "rb").read()
    return [json.loads(line) for line in io.BytesIO(data) if line.strip()]


def load(name):
    s = SETS[name]
    qrels = {}
    for r in jsonl(fetch(s["qrels"])):
        if float(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    queries = [q for q in jsonl(fetch(s["queries"])) if str(q["_id"]) in qrels]
    queries.sort(key=lambda q: str(q["_id"]))
    corpus = jsonl(fetch(s["corpus"]))
    corpus.sort(key=lambda d: str(d["_id"]))
    qtexts = [f"Instruct: {s['instruct']}\nQuery: {q['text']}" for q in queries]
    dtexts = [(d.get("title") or "").strip() + ("\n" if (d.get("title") or "").strip() else "") + d["text"] for d in corpus]
    return dict(qids=[str(q["_id"]) for q in queries], dids=[str(d["_id"]) for d in corpus],
                qtexts=qtexts, dtexts=dtexts, qrels=qrels)


def embed(url, model, texts, conc, batch, trunc):
    out = np.zeros((len(texts), 1536), dtype=np.float32)
    chunks = [(i, texts[i:i + batch]) for i in range(0, len(texts), batch)]
    toks = [0]

    def one(item):
        i, part = item
        body = {"model": model, "input": part, "encoding_format": "base64"}
        if trunc:
            body["truncate_prompt_tokens"] = trunc
        req = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    resp = json.loads(r.read())
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)
        assert len(resp["data"]) == len(part), (len(resp["data"]), len(part))
        for d in resp["data"]:
            out[i + d["index"]] = np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32)
        toks[0] += resp["usage"]["prompt_tokens"]

    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, chunks))
    dt = time.time() - t0
    assert np.isfinite(out).all()
    norms = np.linalg.norm(out, axis=1)
    print(f"embedded {len(texts)} texts, {toks[0]} tokens in {dt:.1f}s ({len(texts) / dt:.0f} texts/s, "
          f"{toks[0] / dt:.0f} tok/s), norm min {norms.min():.6f} max {norms.max():.6f}", flush=True)
    return out


def topk(q, d, k):
    idx = np.empty((q.shape[0], k), dtype=np.int64)
    for s in range(0, q.shape[0], 256):
        sim = q[s:s + 256] @ d.T
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        order = np.take_along_axis(sim, part, 1).argsort(axis=1)[:, ::-1]
        idx[s:s + 256] = np.take_along_axis(part, order, 1)
    return idx


def metrics(ds, emb):
    nq = len(ds["qids"])
    q, d = emb[:nq], emb[nq:]
    top = topk(q, d, 100)
    dids = np.array(ds["dids"])
    ndcg = rec10 = rec100 = mrr = 0.0
    for i, qid in enumerate(ds["qids"]):
        rel = ds["qrels"][qid]
        got = dids[top[i]]
        hits = np.array([x in rel for x in got])
        dcg = (hits[:10] / np.log2(np.arange(2, 12))).sum()
        idcg = (1 / np.log2(np.arange(2, 2 + min(len(rel), 10)))).sum()
        ndcg += dcg / idcg
        rec10 += hits[:10].sum() / len(rel)
        rec100 += hits.sum() / len(rel)
        first = np.flatnonzero(hits[:10])
        mrr += 1 / (first[0] + 1) if first.size else 0
    return dict(ndcg10=ndcg / nq, recall10=rec10 / nq, recall100=rec100 / nq, mrr10=mrr / nq), top


def main():
    cmd = sys.argv[1]
    if cmd == "embed":
        url, model, name, out = sys.argv[2:6]
        conc = int(sys.argv[6]) if len(sys.argv) > 6 else 16
        batch = int(sys.argv[7]) if len(sys.argv) > 7 else 64
        trunc = int(sys.argv[8]) if len(sys.argv) > 8 else 512
        ds = load(name)
        emb = embed(url, model, ds["qtexts"] + ds["dtexts"], conc, batch, trunc)
        np.save(out, emb)
    elif cmd == "score":
        name, files = sys.argv[2], sys.argv[3:]
        ds = load(name)
        nq = len(ds["qids"])
        print(f"{name}: {nq} queries, {len(ds['dids'])} docs")
        base = None
        for f in files:
            emb = np.load(f)
            m, top = metrics(ds, emb)
            line = {"file": os.path.basename(f), **{k: round(v, 5) for k, v in m.items()}}
            if base is None:
                base = (emb, top, m)
            else:
                cos = (emb * base[0]).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(base[0], axis=1))
                ov = np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(top, base[1])])
                t1 = float(np.mean(top[:, 0] == base[1][:, 0]))
                line.update(cos_min=round(float(cos.min()), 6), cos_mean=round(float(cos.mean()), 7),
                            cos_p001=round(float(np.quantile(cos, 0.001)), 6), top10_overlap=round(float(ov), 4),
                            top1_same=round(t1, 4),
                            **{f"d_{k}": round(m[k] - base[2][k], 5) for k in m})
            print(json.dumps(line), flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
