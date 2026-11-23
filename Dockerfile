# giga-serve: embeddings server for ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826, a drop-in
# replacement for the vLLM pooling engine.
#
# The forward reproduces vLLM v0.34.0's eager forward op for op with vLLM's own kernels (cuBLAS
# with vLLM's GEMM shapes, csrc rms_norm / fused_add_rms_norm / silu_and_mul / grouped_topk,
# FlashInfer RoPE on an fp32 cos/sin cache built on the GPU, FA3 varlen non-causal, Triton
# fused_moe with the tuned E=64,N=1280 H200 config): on one batch every layer output is
# bit-equal to vLLM's (tools/layercmp.py). What changes is the serving path: a token-budget
# batcher in one process, CUDA graphs for steps up to 4096 tokens, no per-step EngineCore
# overhead. See giga_serve.py's docstring for the HTTP contract.
#
# Base = the vLLM image whose kernels it calls, pinned by digest (linux/amd64). Those are private
# APIs (vllm._custom_ops, fused_moe.fused_experts, vllm_flash_attn): a base bump needs
# tools/layercmp.py on a GPU (bit-equal taps) before it ships.
#
# Build (no GPU needed, the CPU tests run in the build):
#   docker build --platform linux/amd64 -t giga-serve .
FROM vllm/vllm-openai:v0.34.0@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=true

COPY giga_model.py giga_engine.py giga_serve.py test_giga_serve.py /app/
COPY tools/layercmp.py tools/bench.py tools/drain_test.py /app/tools/

# Imports the server needs, all from the base; then the CPU tests (model math vs transformers'
# DeepseekV3Model made non-causal, request parsing, truncation, every response encoding).
RUN python3 -c "import orjson, uvicorn, uvloop, httptools, prometheus_client, tokenizers, safetensors, flashinfer, vllm; \
assert vllm.__version__ == '0.34.0', vllm.__version__" \
    && cd /app && python3 -m py_compile giga_model.py giga_engine.py giga_serve.py \
    && python3 test_giga_serve.py \
    && rm test_giga_serve.py

LABEL org.opencontainers.image.title="giga-serve" \
      org.opencontainers.image.source="https://github.com/arynyklas/giga-serve" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Embeddings server for ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826@063b0c487fbcde25994c0cbd1d3b17c386e016a6"

WORKDIR /app
EXPOSE 8017
ENTRYPOINT ["python3", "/app/giga_serve.py"]
