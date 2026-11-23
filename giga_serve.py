"""giga-serve: OpenAI-compatible embeddings server for ai-sage/Giga-Embeddings-instruct-10B-A1.8B.

Drop-in for a vLLM pooling engine serving this model (vLLM `--runner pooling --convert embed
--pooler-config {"pooling_type":"MEAN","use_activation":true}`): same HTTP contract on the
embeddings paths, same metric names, same vectors up to bf16 batch noise.

  POST /v1/embeddings  input: str | [str] | [int] | [[int]]; encoding_format float | base64 |
                       bytes | bytes_only (`bytes`: raw vectors + a `metadata`
                       header); embed_dtype, endianness, truncate_prompt_tokens, truncation_side,
                       add_special_tokens, use_activation; `dimensions` -> 400 (no Matryoshka)
  POST /tokenize       {"prompt": str, "add_special_tokens": bool}
  GET  /v1/models, /health, /metrics

One process: uvicorn (httptools, uvloop) on the main thread, tokenisation on a thread pool, the
GPU in giga_engine's thread. The port opens only after the weights are loaded, the CUDA graphs
captured and the eager sizes warmed. SIGTERM stops accepting, finishes every accepted request
(up to --shutdown-timeout), then exits. If the engine thread fails the process exits non-zero.
"""
import argparse
import asyncio
import base64
import concurrent.futures
import json
import os
import signal
import sys
import threading
import time
import uuid

import numpy as np
import orjson

ENCODINGS = ("float", "base64", "bytes", "bytes_only")
EMBED_DTYPES = ("float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2")
ENDIANNESS = ("native", "big", "little")
FIELDS = {"model", "input", "encoding_format", "embed_dtype", "endianness", "dimensions", "user",
          "truncate_prompt_tokens", "truncation_side", "add_special_tokens", "use_activation",
          "request_id", "priority", "padding", "mm_processor_kwargs", "cache_salt"}
MAX_BODY = 64 << 20


class HTTPError(Exception):
    def __init__(self, code, message, typ="BadRequestError", param=None):
        super().__init__(message)
        self.code, self.message, self.typ, self.param = code, message, typ, param

    def body(self):
        return orjson.dumps({"error": {"message": self.message, "type": self.typ, "param": self.param,
                                       "code": self.code}})


def bad(message, param=None):
    return HTTPError(400, message, "BadRequestError", param)


# ---- request parsing (pure; unit-tested) -------------------------------------------------------
def parse_embedding_request(req, served, dim_supported=False):
    """Validates an /v1/embeddings body. Returns a dict of normalised options."""
    if not isinstance(req, dict):
        raise bad("request body must be a JSON object")
    if "normalize" in req:
        raise bad("Parameter `normalize` was removed; use `use_activation` instead. (parameter=normalize)", "normalize")
    model = req.get("model")
    if model is not None and model != served:
        raise HTTPError(404, f"The model `{model}` does not exist.", "NotFoundError")
    if "input" not in req:
        raise bad("Field required: input", "input")
    inp = req["input"]
    if isinstance(inp, str):
        texts, ids = [inp], None
    elif isinstance(inp, list) and inp and all(isinstance(x, str) for x in inp):
        texts, ids = inp, None
    elif isinstance(inp, list) and inp and all(isinstance(x, int) and not isinstance(x, bool) for x in inp):
        texts, ids = None, [inp]
    elif isinstance(inp, list) and inp and all(
            isinstance(x, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in x) for x in inp):
        texts, ids = None, inp
    elif isinstance(inp, list) and not inp:
        raise bad("You must pass at least one prompt")
    else:
        raise bad("input must be a string, a list of strings, a list of token ids or a list of token id lists", "input")
    fmt = req.get("encoding_format", "float")
    if fmt not in ENCODINGS:
        raise bad(f"encoding_format: Input should be {', '.join(repr(x) for x in ENCODINGS)}", "encoding_format")
    dtype = req.get("embed_dtype", "float32")
    if dtype not in EMBED_DTYPES:
        raise bad(f"embed_dtype: Input should be {', '.join(repr(x) for x in EMBED_DTYPES)}", "embed_dtype")
    endian = req.get("endianness", "native")
    if endian not in ENDIANNESS:
        raise bad(f"endianness: Input should be {', '.join(repr(x) for x in ENDIANNESS)}", "endianness")
    if req.get("dimensions") is not None and not dim_supported:
        raise bad(f"Model '{served}' does not support Matryoshka embeddings; dimensions must be unset "
                  f"(received dimensions={req['dimensions']}).")
    trunc = req.get("truncate_prompt_tokens")
    if trunc is not None and (not isinstance(trunc, int) or isinstance(trunc, bool) or trunc < -1):
        raise bad("truncate_prompt_tokens: Input should be greater than or equal to -1", "truncate_prompt_tokens")
    side = req.get("truncation_side") or "right"
    if side not in ("left", "right"):
        raise bad("truncation_side: Input should be 'left' or 'right'", "truncation_side")
    add_special = req.get("add_special_tokens", True)
    use_act = req.get("use_activation")
    for name, val in (("add_special_tokens", add_special), ("use_activation", use_act)):
        if val is not None and not isinstance(val, bool):
            raise bad(f"{name}: Input should be a valid boolean", name)
    return dict(texts=texts, ids=ids, fmt=fmt, dtype=dtype, endian=endian, trunc=trunc, side=side,
                add_special=bool(add_special), normalize=use_act is not False)


def truncate(ids, k, side):
    """vLLM's _token_truncation: keep the first (right) or last (left) k tokens, special tokens included.
    Right truncation equals HF's (BOS + the first k-1 tokens); an explicit left one drops BOS."""
    if k is None or len(ids) <= k:
        return ids
    if k == 0:
        return []
    return ids[-k:] if side == "left" else ids[:k]


def too_long(max_len):
    n = max_len + 1
    return bad(f"This model's maximum context length is {max_len} tokens. However, you requested 0 output tokens "
               f"and your prompt contains at least {n} input tokens, for a total of at least {n} tokens. Please reduce "
               f"the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, "
               f"value={n})")


def encode_vector(vec, dtype, endian):
    """fp32 vector -> bytes in embed_dtype / endianness, as vLLM's tensor2binary."""
    if dtype == "float32":
        arr = vec.astype("<f4" if endian != "big" else ">f4", copy=False)
        return arr.tobytes()
    if dtype == "float16":
        return vec.astype("<f2" if endian != "big" else ">f2").tobytes()
    import torch

    t = torch.from_numpy(np.ascontiguousarray(vec))
    if dtype == "bfloat16":
        raw = t.to(torch.bfloat16).view(torch.int16).numpy()
        return raw.astype(">i2").tobytes() if endian == "big" else raw.tobytes()
    tt = torch.float8_e4m3fn if dtype == "fp8_e4m3" else torch.float8_e5m2
    return t.to(tt).view(torch.uint8).numpy().tobytes()


def build_response(vecs, opts, rid, created, served, ntok):
    """-> (status, headers, body) for float / base64 / bytes / bytes_only."""
    usage = {"prompt_tokens": ntok, "total_tokens": ntok, "completion_tokens": 0,
             "prompt_tokens_details": None, "completion_tokens_details": None}
    fmt = opts["fmt"]
    if fmt == "float":
        data = [{"index": i, "object": "embedding", "embedding": vecs[i]} for i in range(len(vecs))]
        body = orjson.dumps({"id": rid, "object": "list", "created": created, "model": served, "data": data,
                             "usage": usage}, option=orjson.OPT_SERIALIZE_NUMPY)
        return 200, [(b"content-type", b"application/json")], body
    chunks = [encode_vector(vecs[i], opts["dtype"], opts["endian"]) for i in range(len(vecs))]
    if fmt == "base64":
        data = [{"index": i, "object": "embedding", "embedding": base64.b64encode(c).decode()}
                for i, c in enumerate(chunks)]
        body = orjson.dumps({"id": rid, "object": "list", "created": created, "model": served, "data": data,
                             "usage": usage})
        return 200, [(b"content-type", b"application/json")], body
    headers = [(b"content-type", b"application/octet-stream")]
    if fmt == "bytes":
        items, off = [], 0
        for i, c in enumerate(chunks):
            items.append({"index": i, "embed_dtype": opts["dtype"], "endianness": opts["endian"],
                          "start": off, "end": off + len(c), "shape": [len(vecs[i])]})
            off += len(c)
        meta = {"id": rid, "created": created, "model": served, "data": items,
                "usage": {"prompt_tokens": ntok, "total_tokens": ntok}}
        headers.append((b"metadata", json.dumps(meta).encode()))
    return 200, headers, b"".join(chunks)


# ---- metrics ----------------------------------------------------------------------------------
class Metrics:
    """vLLM metric names and labels ({engine="0", model_name=...}) so dashboards and alerts keep working."""

    TIME_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0,
                    240.0, 480.0, 960.0, 1920.0, 7680.0)

    def __init__(self, served):
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = r = CollectorRegistry()
        lab = dict(engine="0", model_name=served)
        names = list(lab)
        self.success = Counter("vllm:request_success", "Count of successfully processed requests.",
                               names + ["finished_reason"], registry=r)
        self.ok = self.success.labels(**lab, finished_reason="stop")
        self.abort = self.success.labels(**lab, finished_reason="abort")
        self.error = self.success.labels(**lab, finished_reason="error")
        self.prompt_tokens = Counter("vllm:prompt_tokens", "Number of prefill tokens processed.", names,
                                     registry=r).labels(**lab)
        Counter("vllm:num_preemptions", "Cumulative number of preemption from the engine.", names,
                registry=r).labels(**lab)
        self.running = Gauge("vllm:num_requests_running", "Number of requests in model execution batches.",
                             names, registry=r).labels(**lab)
        self.waiting = Gauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", names,
                             registry=r).labels(**lab)
        Gauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", names,
              registry=r).labels(**lab).set(0)
        self.iter_tokens = Histogram("vllm:iteration_tokens_total", "Histogram of number of tokens per engine_step.",
                                     names, buckets=(1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
                                     registry=r).labels(**lab)
        self.req_prompt = Histogram("vllm:request_prompt_tokens", "Number of prefill tokens processed.", names,
                                    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000), registry=r).labels(**lab)
        self.e2e = Histogram("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.",
                             names, buckets=self.TIME_BUCKETS, registry=r).labels(**lab)
        self.queue = Histogram("vllm:request_queue_time_seconds",
                               "Histogram of time spent in WAITING phase for request.", names,
                               buckets=self.TIME_BUCKETS, registry=r).labels(**lab)
        self.infer = Histogram("vllm:request_inference_time_seconds",
                               "Histogram of time spent in RUNNING phase for request.", names,
                               buckets=self.TIME_BUCKETS, registry=r).labels(**lab)
        self.step_seconds = Histogram("giga_serve_step_duration_seconds", "GPU step wall time (H2D to D2H).",
                                      ["mode"], buckets=(0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28),
                                      registry=r)
        self.http = Counter("giga_serve_http_requests", "HTTP requests by path and status.", ["path", "code"],
                            registry=r)
        self._steps = {"graph": self.step_seconds.labels(mode="graph"), "eager": self.step_seconds.labels(mode="eager")}
        self.window = [0, 0, 0]  # steps, tokens, seqs since the last log line

    def step(self, tokens, seconds, mode):
        self.iter_tokens.observe(tokens)
        self._steps[mode].observe(seconds)
        self.window[0] += 1
        self.window[1] += tokens

    def aborted(self, n):
        self.abort.inc(n)

    def finished(self, job):
        now = time.monotonic()
        n = len(job.seqs)
        self.ok.inc(n)
        self.prompt_tokens.inc(job.ntok)
        self.window[2] += n
        start = job.t_start or now
        for s in job.seqs:
            self.req_prompt.observe(len(s))
        for _ in range(n):  # vLLM observes these once per prompt (one engine request per input)
            self.e2e.observe(now - job.t_arrive)
            self.queue.observe(start - job.t_arrive)
            self.infer.observe(now - start)

    def render(self, engine):
        from prometheus_client import generate_latest

        self.running.set(engine.running_seqs)
        self.waiting.set(engine.queued_seqs)
        return generate_latest(self.registry)


# ---- server -----------------------------------------------------------------------------------
class Server:
    def __init__(self, args, tokenizer, engine, metrics):
        self.args, self.tok, self.engine, self.metrics = args, tokenizer, engine, metrics
        self.served = args.served_model_name
        self.max_len = args.max_model_len
        self.vocab = engine.m.cfg.vocab
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.tokenizer_threads,
                                                          thread_name_prefix="tok")
        self.created = int(time.time())

    # -- helpers
    async def read_body(self, receive):
        chunks, size = [], 0
        while True:
            m = await receive()
            if m["type"] == "http.disconnect":
                return None
            b = m.get("body", b"")
            size += len(b)
            if size > MAX_BODY:
                raise HTTPError(413, "request body too large", "BadRequestError")
            chunks.append(b)
            if not m.get("more_body"):
                return b"".join(chunks)

    @staticmethod
    async def send(send, status, body, headers=((b"content-type", b"application/json"),)):
        hdrs = list(headers) + [(b"content-length", str(len(body)).encode())]
        await send({"type": "http.response.start", "status": status, "headers": hdrs})
        await send({"type": "http.response.body", "body": body})

    def tokenize(self, texts, add_special, trunc, side):
        encs = self.tok.encode_batch(texts, add_special_tokens=add_special)
        k = self.max_len if trunc == -1 else trunc
        return [truncate(e.ids, k, side) for e in encs]

    # -- endpoints
    async def embeddings(self, receive, send):
        raw = await self.read_body(receive)
        if raw is None:
            return 499
        try:
            req = orjson.loads(raw)
        except orjson.JSONDecodeError as e:
            raise bad(f"JSON decode error: {e}", "body") from None
        opts = parse_embedding_request(req, self.served)
        trunc = opts["trunc"]
        if trunc is not None and trunc > self.max_len:
            raise bad("truncate_prompt_tokens value is greater than max_model_len. Please request a smaller "
                      "truncation size.")
        if opts["texts"] is not None:
            texts = opts["texts"]
            if sum(len(t) for t in texts) > 4096:
                ids = await asyncio.get_running_loop().run_in_executor(
                    self.pool, self.tokenize, texts, opts["add_special"], trunc, opts["side"])
            else:
                ids = self.tokenize(texts, opts["add_special"], trunc, opts["side"])
        else:
            k = self.max_len if trunc == -1 else trunc
            ids = [truncate(list(x), k, opts["side"]) for x in opts["ids"]]
            for x in ids:
                for t in x:
                    if t < 0 or t >= self.vocab:
                        raise bad(f"Token id {t} is out of vocabulary")
        for x in ids:
            if len(x) > self.max_len:
                raise too_long(self.max_len)
            if not x:
                raise bad("The prompt is empty: send at least one token")
        seqs = [np.asarray(x, dtype=np.int32) for x in ids]
        ntok = int(sum(len(s) for s in seqs))
        eng = self.engine
        # Backpressure only against what is already queued: a request bigger than the limit
        # on its own still runs when the queue is empty (a 503 there could never succeed).
        if eng.queued_tokens and eng.queued_tokens + ntok > self.args.max_queued_tokens:
            raise HTTPError(503, f"server busy: {eng.queued_tokens} tokens queued", "ServiceUnavailableError")
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def done(job, err):
            loop.call_soon_threadsafe(_resolve, fut, job, err)

        from giga_engine import Job

        job = Job(seqs, eng.dim, done)
        eng.submit(job)
        watch = asyncio.ensure_future(receive())  # resolves on http.disconnect
        try:
            finished, _ = await asyncio.wait({fut, watch}, return_when=asyncio.FIRST_COMPLETED)
            if fut not in finished:
                eng.cancel(job)
                return 499
        finally:
            if not watch.done():
                watch.cancel()
        err = fut.result()
        if err is not None:
            raise HTTPError(500, f"engine failed: {err!r}", "InternalServerError")
        self.metrics.finished(job)
        vecs = job.out if opts["normalize"] else job.raw
        rid = f"embd-{uuid.uuid4().hex}"
        status, headers, body = build_response(vecs, opts, rid, int(time.time()), self.served, ntok)
        await self.send(send, status, body, headers)
        return status

    async def tokenize_ep(self, receive, send):
        raw = await self.read_body(receive)
        if raw is None:
            return 499
        try:
            req = orjson.loads(raw)
        except orjson.JSONDecodeError as e:
            raise bad(f"JSON decode error: {e}", "body") from None
        if not isinstance(req, dict) or not isinstance(req.get("prompt"), str):
            raise bad("only {\"prompt\": str} is supported", "prompt")
        model = req.get("model")
        if model is not None and model != self.served:
            raise HTTPError(404, f"The model `{model}` does not exist.", "NotFoundError")
        ids = self.tok.encode(req["prompt"], add_special_tokens=req.get("add_special_tokens", True) is not False).ids
        await self.send(send, 200, orjson.dumps({"count": len(ids), "max_model_len": self.max_len, "tokens": ids,
                                                 "token_strs": None}))
        return 200

    def models_body(self):
        return orjson.dumps({"object": "list", "data": [{
            "id": self.served, "object": "model", "created": self.created, "owned_by": "giga-serve",
            "root": self.args.model, "parent": None, "max_model_len": self.max_len, "permission": []}]})

    async def app(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                m = await receive()
                if m["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif m["type"] == "lifespan.shutdown":
                    self.engine.stop()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        code = 500
        try:
            if path == "/v1/embeddings":
                if method != "POST":
                    code = 405
                    await self.send(send, 405, b'{"detail":"Method Not Allowed"}')
                else:
                    code = await self.embeddings(receive, send)
            elif path == "/health":
                code = 200 if self.engine.dead is None else 503
                await self.send(send, code, b"", ())
            elif path == "/metrics":
                code = 200
                await self.send(send, 200, self.metrics.render(self.engine),
                                ((b"content-type", b"text/plain; version=0.0.4; charset=utf-8"),))
            elif path == "/v1/models" and method == "GET":
                code = 200
                await self.send(send, 200, self.models_body())
            elif path == "/tokenize" and method == "POST":
                code = await self.tokenize_ep(receive, send)
            else:
                code = 404
                await self.send(send, 404, b'{"detail":"Not Found"}')
        except HTTPError as e:
            code = e.code
            if e.code >= 500:
                self.metrics.error.inc()
            await self.send(send, e.code, e.body())
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            code = 500
            self.metrics.error.inc()
            await self.send(send, 500, HTTPError(500, repr(e), "InternalServerError").body())
        finally:
            if path in ("/v1/embeddings", "/tokenize"):
                self.metrics.http.labels(path=path, code=str(code)).inc()


def _resolve(fut, job, err):
    if not fut.done():
        fut.set_result(err)


# ---- startup ----------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--served-model-name", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8017)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384, help="tokens per GPU step")
    p.add_argument("--max-num-seqs", type=int, default=512, help="sequences per GPU step")
    p.add_argument("--graph-max-tokens", type=int, default=4096,
                   help="steps up to this many tokens replay CUDA graphs; bigger steps run eagerly (0 = never)")
    p.add_argument("--max-queued-tokens", type=int, default=8 << 20, help="503 above this queue depth")
    p.add_argument("--tokenizer-threads", type=int, default=4)
    p.add_argument("--shutdown-timeout", type=int, default=600, help="seconds to finish accepted requests")
    p.add_argument("--log-interval", type=float, default=10.0)
    return p.parse_args(argv)


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} giga-serve {msg}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    # tokenizers' encode_batch runs on rayon's global pool, one thread per host CPU by default
    # (hundreds on a large GPU host): keep it to the process's CPU budget, like the torch threads.
    os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))
    import torch
    from tokenizers import Tokenizer

    import giga_model as G
    from giga_engine import Engine

    if not torch.cuda.is_available():
        sys.exit("giga-serve needs a CUDA device")
    t0 = time.time()
    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    cfg = G.load_cfg(args.model)
    dev = torch.device("cuda:0")
    model = G.GigaEmbedModel(cfg, dev, max_pos=args.max_model_len)
    log(f"weights loaded in {model.load(args.model):.1f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
        f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    metrics = Metrics(args.served_model_name)

    def on_fatal(err):
        log(f"FATAL engine error, exiting: {err!r}")
        threading.Timer(1.0, lambda: os._exit(1)).start()

    eng = Engine(model, max_tokens=args.max_num_batched_tokens, max_seqs=args.max_num_seqs,
                 max_len=args.max_model_len, graph_max_tokens=args.graph_max_tokens, metrics=metrics,
                 on_fatal=on_fatal)
    if eng.t_buckets:
        n, dt = eng.capture()
        log(f"captured {n} CUDA graphs in {dt:.1f}s")
    n, dt = eng.warmup_eager()
    log(f"warmed {n} eager step sizes in {dt:.1f}s; GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB "
        f"reserved; ready after {time.time() - t0:.0f}s")
    eng.start()
    server = Server(args, tok, eng, metrics)

    def stats_loop():
        last = time.monotonic()
        while True:
            time.sleep(args.log_interval)
            now = time.monotonic()
            steps, toks, seqs = metrics.window
            metrics.window = [0, 0, 0]
            if steps:
                dt = now - last
                log(f"{seqs / dt:.0f} inputs/s, {toks / dt:.0f} tok/s, {steps / dt:.1f} steps/s, "
                    f"waiting {eng.queued_seqs}")
            last = now

    threading.Thread(target=stats_loop, daemon=True, name="stats").start()
    import uvicorn

    config = uvicorn.Config(server.app, host=args.host, port=args.port, loop="uvloop", http="httptools",
                            lifespan="on", interface="asgi3", access_log=False, log_level="warning", backlog=4096,
                            timeout_keep_alive=75, timeout_graceful_shutdown=args.shutdown_timeout)
    log(f"serving {args.served_model_name} on {args.host}:{args.port}")
    uvicorn.Server(config).run()
    log("stopped")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
