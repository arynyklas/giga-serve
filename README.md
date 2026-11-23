# giga-serve

An OpenAI-compatible embeddings server for
[ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826)
(a bidirectional DeepSeek-V3-style MoE, 10B total / 1.8B active, 1536-dim vectors, mean pooling).

It replaces a vLLM pooling engine for this one model. The forward pass calls **vLLM v0.34.0's own
kernels in vLLM's exact order**, so on the same batch every layer output is bit-identical to vLLM's
(vectors too, up to the order of fp32 atomics in the mean pool). Only the serving path is
new: one process, one token-budget batcher, and CUDA graphs. The result is ~1.05x vLLM's throughput
on batched requests, **1.45–3x on single-text requests**, and 4.4x lower single-request latency.

Validated on one **NVIDIA H200**.

## Why

vLLM spends roughly 55 ms of CPU per EngineCore step on this model at any batch size. Batched clients
hide that cost; many concurrent single-text clients wait on it. giga-serve replays a pre-captured CUDA
graph per step (~0.5 ms of CPU), so small requests are no longer bound by the scheduler.

## Results (one H200, same client and inputs for both engines)

Throughput in inputs/s, `bytes` encoding, two runs each. 40-word texts are ~61 tokens, 320-word
texts ~480 tokens (`tools/bench.py`, seed 7).

| Shape (concurrent requests x texts per request) | vLLM v0.34.0 (eager, 4 API servers) | giga-serve | ratio |
|---|---:|---:|---:|
| 16 x 64 | 2,112 / 2,179 | 2,275 / 2,266 | 1.05x |
| 32 x 64 | 2,189 / 2,182 | 2,268 / 2,262 | 1.04x |
| 16 x 64, 320 words | 271.9 / 272.1 | 275.0 / 274.6 | 1.01x (GPU-bound) |
| 64 x 1 | 959 / 1,002 | 1,481 / 1,421 | 1.45x |
| 16 x 1 | 272.5 / 293.1 | 861 / 858 | 3.0x |
| 1 x 1, p50 latency | 31.1 ms | 7.0 ms | 4.4x |

- Batched requests ran at ~139k tok/s in this comparison, about 95% of the raw forward ceiling of that
  version (~146k tok/s at 16k-token steps). The optimizations below raised both by ~7%.
- Startup: ready 8–11 s after launch (weights from the page cache plus the CUDA graphs), against
  ~55 s for vLLM.

### Correctness

| Check | Result |
|---|---|
| Per-layer comparison against vLLM v0.34.0 in-process (`tools/layercmp.py`: one 64-sequence step of 4,003 tokens, forward hooks on every layer) | all 52 layer outputs (26 attention, 26 MLP) and every sub-op of layers 0–1 **bit-equal**; 63/64 vectors bit-equal, the 64th cos 0.9999998 (order of fp32 atomics in the mean pool) |
| Retrieval against vLLM 0.28 serving the same model (`tools/retrieval_eval.py`: SciFact 300 queries / 5,183 docs, RuBQ 1,692 / 56,826) | nDCG@10 0.75978 / 0.80926, recall@10 0.87933 / 0.92358, recall@100 0.96333 / 0.98887, MRR@10 0.72679 / 0.81847: **identical to 5 decimals**. Top-10 overlap 1.0, top-1 unchanged for 100% of queries. Per-vector min cos 0.99965 / 0.99944, the same as vLLM's own run-to-run spread |
| HTTP contract, 41 cases against vLLM | same status and content type in all 41, same `usage.prompt_tokens`, same `/tokenize` ids, same `bytes` metadata |
| `truncate_prompt_tokens` (right and left, text and token ids) against vLLM v0.34.0 | cos 1.0000000 |

The detail that made the match exact: vLLM builds the YaRN cos/sin cache **on the GPU**. The same
formula computed on the CPU differs by one ulp in a few entries. That is enough to shift layer-0
attention and drop the mean cos to 0.9994.

Like vLLM, a vector still depends slightly on which texts share its GPU step (bf16 GEMM and MoE tile
choices depend on the token count). Outputs are not batch-invariant.

### Optimizations after the first release (same H200; no change on the correctness checks below)

Measured with the client in the same container as the server, so absolute numbers are higher than in
the table above (no network); previous build vs current build, side by side, two to three runs each.

| Shape (inputs/s) | before | after | gain |
|---|---:|---:|---:|
| 16 x 64 | 2,646 / 2,641 / 2,633 | 2,850 / 2,848 / 2,847 | +7.7% |
| 32 x 64 | 2,643 / 2,637 | 2,835 / 2,839 | +7.4% |
| 16 x 64, 320 words | 326.2 / 326.2 | 347.7 / 349.0 | +6.8% |
| 1 x 1 (p50) | 146.4 / 146.8 (6.7 ms) | 151.6 / 152.6 (6.5 ms) | +3.7% |
| raw forward, 16k-token step | 141k tok/s | 151k tok/s | +7.2% |

What changed, from a kernel profile of a 16k-token step:

- **A hidden copy per layer.** Debug-tap arguments were evaluated even with tapping off, and one of
  them reshaped the strided V view, i.e. copied 16k x 6144 bf16 per layer (6 ms per step).
- **K assembly in one Triton kernel** (`_k_assemble`) instead of two strided slice copies
  (6.4 ms -> 2.3 ms per step). Pure copies, so the values are identical.
- **Tuned `fused_moe` tile configs** for this model's shapes at the server's default step sizes
  (`moe_configs/`, produced by `tools/tune_moe.py --ms server`); other step sizes use the nearest
  tuned size, as vLLM does. Every config in the file was `torch.equal` to vLLM's shipped config on
  the tuning inputs (real routing of one MoE layer) and on whole-model hidden states at 6 step sizes.
  The gain is 3-14% on the MoE kernel for 32-768-token steps, 0-5% above.
- **Finer CUDA-graph buckets** above 512 tokens (at most 1.25x apart, 100 graphs instead of 64):
  +2-3.5% for 12-28 concurrent single-text clients; startup +1.7 s, +0.4 GB. A step is now padded to
  a different size than before, so, like any change of batch shape, its vectors can differ in the last
  bits (see the batch-invariance note above).

Checks on the final build: `tools/layercmp.py` PASS (52/52 layer outputs bit-equal to vLLM v0.34.0,
64/64 vectors); the hidden states of 6 step sizes (48 to 16,384 tokens) are bit-identical with the tuned
MoE configs and with vLLM's shipped ones; RuBQ and SciFact metrics identical to the previous build
(min per-vector cos 1.0); a small real-world FAQ set (7 short Russian articles, 16 labelled
questions, cosine-threshold decisions) gives identical decisions and |Δcos| = 0.

Tried and dropped:

- `torch._grouped_mm` (CUTLASS grouped GEMM) instead of Triton `fused_experts`: 1.10x at 16k-token
  steps, 0.75-0.8x below 2k, and the vectors change (cos ~0.99999). Not worth it.
- `--max-num-batched-tokens 32768`: +1.4% throughput for twice the step latency; the default stays 16384.
- A segmented mean-pool kernel: pooling is 0.07 ms of a 108 ms step.

## How it works

- `giga_model.py`: the forward pass.
  - Two cuBLAS GEMMs for `q_proj` / `kv_a_proj_with_mqa`, in vLLM's shapes, so cuBLAS picks vLLM's
    algorithms.
  - vLLM's csrc kernels: `rms_norm`, `fused_add_rms_norm`, `silu_and_mul`, `grouped_topk`.
  - FlashInfer RoPE on an fp32 cos/sin cache.
  - FA3 varlen non-causal attention.
  - Triton `fused_experts` with tile configs tuned for this model on H200 (`moe_configs/`; other GPUs
    fall back to vLLM's shipped configs).
  - Mean pooling in fp32, then `F.normalize`.
  - Without CUDA, every op falls back to plain torch; the CPU unit tests use this path.
- `giga_engine.py`: the batcher and GPU loop.
  - A FIFO token-budget batcher: 16,384 tokens / 512 sequences per step by default; one request
    may span several steps.
  - Steps up to 4,096 tokens replay one of 100 CUDA graphs (token bucket x max-length bucket).
    Bigger steps run eagerly, where GPU time hides the kernel launches.
  - A second step is launched behind the running one only when the queue already holds a full
    step. Light load keeps large batches, and a saturated GPU never idles between steps.
- `giga_serve.py`: the HTTP layer, metrics, startup and shutdown.
  - uvicorn (httptools + uvloop) on the main thread, tokenization on a thread pool, the GPU in the
    engine thread.

## API

- `POST /v1/embeddings`:
  - `input`: `str`, `[str]`, `[int]` or `[[int]]`.
  - `encoding_format`: `float`, `base64`, `bytes` or `bytes_only`. `bytes` returns raw vectors
    plus a `metadata` header, as in vLLM.
  - Also accepts `embed_dtype`, `endianness`, `truncate_prompt_tokens`, `truncation_side`,
    `add_special_tokens`, `use_activation`.
  - `dimensions` returns 400: the model has no Matryoshka support.
- `POST /tokenize`, `GET /v1/models`, `GET /health`.
- `GET /metrics` uses vLLM's metric names and labels, so existing dashboards and alerts keep working:
  - `vllm:request_success_total{finished_reason}`, `vllm:prompt_tokens_total`,
    `vllm:num_requests_running|waiting`, `vllm:iteration_tokens_total`, and the latency histograms,
    labelled `{engine="0",model_name=...}`;
  - plus `giga_serve_step_duration_seconds{mode}` and `giga_serve_http_requests_total{path,code}`.

Instructions follow the model card: queries are sent as `Instruct: <task>\nQuery: <text>`, documents
as plain text.

## Run

Requires a Hopper GPU (FA3), i.e. H100 or H200; tested only on H200. The weights take ~20 GB in bf16.

```sh
# weights (pinned revision used for validation)
huggingface-cli download ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826 \
  --revision 063b0c487fbcde25994c0cbd1d3b17c386e016a6 --local-dir ./giga

docker build --platform linux/amd64 -t giga-serve .
docker run --gpus '"device=0"' --ipc=host -p 8017:8017 -v "$PWD/giga:/model:ro" \
  -e OMP_NUM_THREADS=8 giga-serve \
  --model /model --served-model-name giga-embeddings \
  --max-model-len 2048 --max-num-batched-tokens 16384 --max-num-seqs 512
```

```sh
curl -s localhost:8017/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"giga-embeddings","input":["Instruct: Given a question, retrieve passages that answer it\nQuery: What is the capital of France?","Paris is the capital of France."]}'
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-model-len` | 2048 | longer inputs get vLLM's 400, unless `truncate_prompt_tokens` is set |
| `--max-num-batched-tokens` | 16384 | tokens per GPU step |
| `--max-num-seqs` | 512 | sequences per GPU step |
| `--graph-max-tokens` | 4096 | steps up to this size replay CUDA graphs (0 = always eager) |
| `--max-queued-tokens` | 8M | queue depth above which new requests get 503; a request alone on an empty queue is always accepted |
| `--tokenizer-threads` | 4 | tokenization pool size |
| `--shutdown-timeout` | 600 | seconds to finish accepted requests after SIGTERM |

The port opens only after the weights are loaded, the CUDA graphs captured and the eager sizes warmed
up, so `/health` doubles as a readiness probe.

## Failure behaviour

Each case was checked on the H200:

- **SIGTERM:** stops accepting new connections, answers every accepted request, then exits. With
  32 x 64 x ~470-token requests in flight, all 32 returned 200 and the process exited 7.8 s after
  the signal (`tools/drain_test.py`).
- **Client disconnect:** the request's sequences not yet on the GPU are dropped and counted as
  `finished_reason="abort"`.
- **Engine-thread exception:** every in-flight request gets 500 and the process exits with status 1
  (~1.7 s), so a supervisor restarts it. vLLM does the same when its EngineCore dies.

## Tests and tools

- `test_giga_serve.py` runs on CPU and is executed during `docker build`:
  - a tiny DeepSeek-V3 checked against transformers' `DeepseekV3Model` made non-causal;
  - request parsing, truncation, every response encoding, and the `bytes` metadata contract.
- `tools/layercmp.py MODEL_DIR [N]`: per-layer bit comparison against vLLM in-process. Needs a GPU.
  Exits 1 if any layer output differs, so it works as a gate.
- `tools/bench.py URL MODEL CONC BATCH NREQ OUT.npy [WORDS] [ENC]`: throughput and a vector dump.
- `tools/retrieval_eval.py embed|score ...`: retrieval A/B (nDCG, recall, MRR, per-vector cosine)
  on RuBQ and SciFact.
- `tools/drain_test.py URL MODEL N WORDS`: SIGTERM drain check.
- `tools/tune_moe.py MODEL_DIR OUT.json [--ms server] [--quick]`: tunes the `fused_moe` tile configs
  on the local GPU and reports which candidates are bit-identical to vLLM's shipped config. Copy
  `OUT.bitexact.json` into `moe_configs/` under the file name it prints (e.g. for an H100).

The three server modules are fully annotated and pass `mypy --strict` (config in `pyproject.toml`).
vLLM, FlashInfer and the other untyped dependencies are treated as `Any`. To run the checks:

```sh
docker run --rm --entrypoint sh -v "$PWD:/src" -w /src \
  vllm/vllm-openai:v0.34.0@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484 \
  -c 'pip install -q mypy && mypy && python3 test_giga_serve.py'
```

## Upgrading the base image

The server imports private vLLM APIs (`vllm._custom_ops`, `fused_moe.fused_experts`,
`vllm_flash_attn`), so the base image is pinned by digest. Before shipping a new base, run
`tools/layercmp.py` on a GPU and require bit-equal layer taps.

## Possible next steps

These are not done yet. Each one needs `tools/layercmp.py`; the ones marked *numerics* also need the
retrieval A/B, because they change values.

- Copy back only the requested variant (normalised or raw) instead of both.
- Fuse `silu_and_mul` (4% of a step) into the first MoE GEMM's epilogue. *Numerics* unless the bf16
  rounding of the GEMM output is kept.
- Batch-invariant kernels, so a vector would not depend on its batch neighbours. *Numerics.*

## License

Apache-2.0 (see `LICENSE`). The model weights are under their own license; see the model card.
