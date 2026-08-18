# Qwen3.8-27B-GPTQ-Int4 serving on SGLang (with working MTP)

Serve **Qwen3.8-27B GPTQ-Int4** (`Qwen3.8-27B-GPTQ`) with
[SGLang](https://docs.sglang.io/) on NVIDIA GPUs, **with MTP speculative
decoding actually working** (accept rate ~0.97, not 0.00).

This repo is the working configuration we run in-house (4x RTX 3060, x86,
tensor-parallel 4) — adapted from
[MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
which targets DGX Spark (NVFP4) only. The whole point here is the **GPTQ-Int4
path**, where SGLang has a real bug that silently kills MTP (see below).

- Served model name matches the vLLM deployment: **`Qwen3.8-27B-GPTQ`**, so
  existing OpenAI/Anthropic clients keep working without config changes.
- Tested with `lmsysorg/sglang:qwen38-27b` (multi-arch, amd64 + arm64).

## Quick start

```bash
# 1. Edit scripts/start.sh defaults if needed (or override via env):
#    MODEL_PATH, SERVED_MODEL_NAME, PORT, TP_SIZE, MODELS_HOST, MAX_CONCURRENT ...
#    Two ways to point at a checkpoint:
#      - HuggingFace (default of this repo when MODELS_HOST is empty): leave
#        MODELS_HOST unset and set MODEL_PATH to a repo id, e.g.
#        MODEL_PATH=palmfuture/Qwen3.8-27B-GPTQ-Int4  (auto-download, cached)
#      - Local dir: set MODEL_PATH to an absolute path inside the container AND
#        MODELS_HOST to the host directory to bind-mount there.
#    We use: https://huggingface.co/palmfuture/Qwen3.8-27B-GPTQ-Int4 (GPTQ-Int4, 19 GB)

# 2. Start (MTP + the fix are on by default)
./scripts/start.sh

# 3. Verify
curl http://127.0.0.1:8001/v1/models
#    -> data[].id == "Qwen3.8-27B-GPTQ"  (same id as the vLLM deployment)

# 4. Stop
./scripts/stop.sh
```

Environment overrides (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/models/Qwen3.8-27B-GPTQ-Int4` | checkpoint: absolute path inside container, **or a HuggingFace repo id** (e.g. `palmfuture/Qwen3.8-27B-GPTQ-Int4`) — HF id auto-downloads into `HF_CACHE_HOST` |
| `MODELS_HOST` | *(empty)* | host dir mounted at `MODEL_PATH` (`-v`). Only used when `MODEL_PATH` is an absolute path; leave empty to load from HuggingFace |
| `SERVED_MODEL_NAME` | `Qwen3.8-27B-GPTQ` | **must match vLLM** so clients don't change |
| `PORT` | `8001` | host port (host network) |
| `TP_SIZE` | `4` | tensor parallelism |
| `CONTEXT_LENGTH` | `262144` | context window |
| `MAX_CONCURRENT` | `8` | also sizes the GDN mamba state pool (`×4`) |
| `CHUNKED_PREFILL` | `8192` | prefill chunk tokens |
| `SPEC` | `EAGLE` | `EAGLE` (MTP on, default) or `off` |
| `SPEC_STEPS/SPEC_TOPK/SPEC_DRAFT` | `3/1/4` | MTP chain params |
| `HF_CACHE_HOST` / `TRITON_CACHE_HOST` | `~/.cache/sglang_hf` / `~/.triton` | host cache dirs (reused across restarts) |

## The MTP fix (the important part)

**Symptom:** with a GPTQ-Int4 checkpoint you enable MTP via
`--speculative-algorithm EAGLE 3/1/4` and the server boot is fully clean —
`accept len: 1.00, accept rate: 0.00` forever, i.e. every draft token is
rejected. Speed is unchanged (no gain), and you would think MTP "doesn't work"
on your hardware.

**Root cause:** SGLang auto-selects the **`gptq_marlin`** kernel for 4-bit
weighted checkpoints. In `Qwen3_5ForCausalLMMTP` (and the Qwen3.5/3.8 family), the MTP
head weights are **excluded from quantization** (`mtp.*` is a *negative* rule in
the GPTQModel `dynamic` dict, and `mtp.safetensors` stores them as bf16/fp16).
But the draft-model loader applies the `quant_config` to the MTP head anyway →
it decodes bf16 weights through GPTQ/Marlin kernels → garbage drafts → 0%
acceptance. vLLM doesn't hit this because it loads the MTP drafter as a
separate model and shares embed/lm_head weights.

**Fix:** mirror what SGLang already does for ModelOpt / Quark / NPU checkpoints
(see `sgl-project/sglang#23113`): detect that `mtp.*` is excluded from
quantization and pass `quant_config=None` to the MTP head. Applied via a
bind-mounted patch in `patches/qwen3_5_mtp.py`:

```python
# (inside Qwen3_5ForCausalLMMTP.__init__)
if quant_config and quant_config.get_name() in ("gptq", "gptq_marlin"):
    dynamic = getattr(quant_config, "dynamic", None) or {}
    has_mtp_exclusion = any(
        isinstance(k, str) and k.startswith("-") and "mtp" in k
        for k in dynamic
    )
    if has_mtp_exclusion:
        quant_config = None
```

**Result (measured, in-house 4x RTX 3060 / TP4, greedy, E2E non-stream):**

| Engine | essay (512 tok) | code (512 tok) |
|---|---|---|
| vLLM 0.27.1 + MTP | 59.6 tok/s | 84.0 tok/s |
| **SGLang + MTP (this repo)** | **76.6 tok/s** | **101.4 tok/s** |

Before the fix, SGLang sat at ~29 tok/s with `accept rate 0.00`; after the fix
`accept rate 0.97`, mean accept length ~3.9.

> ⚠️ **Benchmarking note:** streaming benchmarks that subtract TTFT can give
> misleadingly low numbers (we initially measured vLLM at ~19.5 tok/s that way).
> Use the non-stream E2E scripts in `bench/` (they count
> `usage.completion_tokens` against wall-clock time) for honest apples-to-apples
> comparisons.

## Files

```
.
├── scripts/
│   ├── start.sh          # launch SGLang container (MTP fix bind-mount included)
│   └── stop.sh
├── patches/
│   └── qwen3_5_mtp.py    # patched Qwen3.5/3.8 MTP head (GPTQ/Marlin fix)
└── bench/
    ├── bench_e2e.py      # non-stream E2E benchmark (recommended; usage-based)
    └── bench_llm.py      # streaming benchmark (for reference only)
```

## Other hard-won flags (RTX 3060 / multi-GPU, GPTQ)

These are all in `start.sh` with comments; the short version of why:

- `--dtype float16` — **required** for GPTQ (SGLang rejects gptq with bf16:
  `torch.bfloat16 is not supported for quantization method gptq`).
- P2P / IPC off for consumer GPUs without NVLink:
  `--disable-custom-all-reduce --weight-cache-mode off`
  `--mm-feature-transport cpu` + env `NCCL_P2P_DISABLE=1`
  `NCCL_CUMEM_ENABLE=0` `SGLANG_USE_IPC_POOL_HANDLE_CACHE=0`
  (the model is a VLM → multimodal features try CUDA IPC by default).
- Small VRAM (12 GB/card here):
  `--mem-fraction-static 0.75 --cuda-graph-max-bs-decode 8`
  + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`.
- GDN / Mamba state pool:
  `--mamba-ssm-dtype float16` **and** env `SGLANG_MAMBA_CONV_DTYPE=float16`
  (the conv state defaults to bf16 which crashes against fp16 activations with
  `Index put requires the source and destination dtypes match, got BFloat16 ... Half`).
  State pool = concurrency × 4 (`extra_buffer_lazy`).
- FP8 KV cache (`--kv-cache-dtype fp8_e4m3`), flashinfer attention,
  prefill CUDA graphs disabled (GDN layers), YaRN off / native 262K context.

## References

- SGLang cookbook Qwen3.8-27B (DGX Spark recipe): https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- MiaAI-Lab repo (DGX Spark version, NVFP4): https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
- SGLang issue #23113 (MT head bf16 + quantization): https://github.com/sgl-project/sglang/issues/23113
- Model card: https://huggingface.co/Qwen/Qwen3.8-27B
- SGLang speculative decoding docs: https://docs.sglang.io/docs/advanced_features/speculative_decoding

## License

MIT
