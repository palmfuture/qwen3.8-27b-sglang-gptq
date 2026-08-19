#!/usr/bin/env bash
# Serve Qwen3.8-27B-GPTQ-Int4 with SGLang on NVIDIA GPUs (tested: 4x RTX 3060, x86).
#
# Adapted from MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark, for GPTQ-Int4 checkpoints
# on multi-GPU (TP) instead of NVFP4-on-DGX-Spark. Key differences:
#   + --dtype float16 (GPTQ requires fp16, not bf16)
#   + TP across GPUs (--tp 4 on our box) with P2P/IPC workarounds (3060 has no P2P)
#   + MTP head fix: we bind-mount a patched qwen3_5_mtp.py so the MTP head is
#     NOT decoded through GPTQ/Marlin kernels (it is stored bf16/fp16).
#   + served model name matches vLLM: "Qwen3.8-27B-GPTQ" (clients unchanged)
set -euo pipefail

# ---- overridable via env ----
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B-GPTQ-Int4}"   # inside container OR HF repo id (e.g. palmpfuture/Qwen3.8-27B-GPTQ-Int4)
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B-GPTQ}"   # matches vLLM exactly
IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="${CONTAINER_NAME:-sglang_qwen38}"
PORT="${PORT:-8001}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
TP_SIZE="${TP_SIZE:-4}"
HF_CACHE_HOST="${HF_CACHE_HOST:-${HOME}/.cache/sglang_hf}"   # reuse across restarts
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-${HOME}/.triton}"
MODELS_HOST="${MODELS_HOST:-}"        # host dir mounted at $MODEL_PATH (only valid when MODEL_PATH is an absolute path)
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# Repo root (this script lives in scripts/ unless STANDALONE=1 copies patch into container)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MTP_PATCH="${REPO_ROOT}/patches/qwen3_5_mtp.py"

command -v docker >/dev/null || { echo "docker missing"; exit 1; }
[[ -f "${MTP_PATCH}" ]] || { echo "MTP patch not found at ${MTP_PATCH}"; exit 1; }

# ---- mamba state pool: concurrency x 4 slots (extra_buffer_lazy, S=4) ----
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT * 4 ))

# ---- speculative decoding (MTP via EAGLE path). SPEC=off disables ----
SPEC="${SPEC:-EAGLE}"
if [[ "${SPEC}" == "off" || "${SPEC}" == "none" || "${SPEC}" == "NONE" ]]; then
  SPEC_ARGS=(--speculative-algorithm NONE)
  echo "Speculative decoding: OFF"
else
  SPEC_STEPS="${SPEC_STEPS:-3}"
  SPEC_TOPK="${SPEC_TOPK:-1}"
  SPEC_DRAFT="${SPEC_DRAFT:-4}"
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps "${SPEC_STEPS}"
    --speculative-eagle-topk "${SPEC_TOPK}"
    --speculative-num-draft-tokens "${SPEC_DRAFT}"
  )
  echo "Spec decode: MTP/EAGLE steps=${SPEC_STEPS} topk=${SPEC_TOPK} draft=${SPEC_DRAFT}"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

mkdir -p "${HF_CACHE_HOST}" "${TRITON_CACHE_HOST}"

MODEL_MOUNT_ARGS=()
if [[ -n "${MODELS_HOST}" ]]; then
  # absolute container path + host dir -> local bind mount
  MODEL_MOUNT_ARGS=(-v "${MODELS_HOST}:${MODEL_PATH}:ro")
else
  # MODEL_PATH is treated as a HuggingFace repo id (downloaded into HF_CACHE_HOST)
  :
fi

echo "Starting SGLang for ${MODEL_PATH} (${SERVED_MODEL_NAME}) tp=${TP_SIZE}"
if [[ -n "${MODELS_HOST}" ]]; then
  echo "Model source: local bind mount ${MODELS_HOST} -> ${MODEL_PATH}"
else
  echo "Model source: HuggingFace repo id (cache: ${HF_CACHE_HOST})"
fi
echo "Context: ${CONTEXT_LENGTH} | concurrent: ${MAX_CONCURRENT} (mamba pool ${MAMBA_CACHE_SIZE})"
echo "MTP head patch: ${MTP_PATCH}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --gpus all \
  --shm-size 32g \
  -e TZ=Asia/Bangkok \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e SGLANG_USE_IPC_POOL_HANDLE_CACHE=0 \
  -e SGLANG_MAMBA_CONV_DTYPE=float16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
  "${MODEL_MOUNT_ARGS[@]}" \
  -v "${MTP_PATCH}:/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py:ro" \
  -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_HOST}:/root/.triton" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype float16 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --weight-cache-mode off \
  --mm-feature-transport cpu \
  --tp "${TP_SIZE}" \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs-decode 8 \
  --chunked-prefill-size "${CHUNKED_PREFILL}" \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype float16 \
  --enable-linear-replayssm-spec \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT}" \
  --context-length "${CONTEXT_LENGTH}" \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false, "reasoning_effort": "low"}' \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --host 0.0.0.0 \
  --port "${PORT}" \
  >/dev/null

cid=$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")
echo "spawned ${cid}"
echo "waiting for readiness at ${READY_URL}..."
for i in $(seq 1 60); do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "CONTAINER EXITED — last log lines:"
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -n 200
    exit 1
  fi
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    echo "SGLang READY at http://0.0.0.0:${PORT}/v1 (model ${SERVED_MODEL_NAME})"
    echo "OpenAI-compatible:  ${READY_URL}"
    echo "Anthropic-compatible: http://0.0.0.0:${PORT}/v1/messages"
    exit 0
  fi
  sleep 10
done
echo "timed out waiting"
exit 1
