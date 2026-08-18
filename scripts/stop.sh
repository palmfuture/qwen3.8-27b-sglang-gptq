#!/usr/bin/env bash
# Stop the SGLang container started by start.sh.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sglang_qwen38}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} does not exist; nothing to stop"
  exit 0
fi

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping container ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null
  echo "Stopped."
else
  echo "Container ${CONTAINER_NAME} is not running"
fi

# Leave the stopped container in place; start.sh removes it on next launch,
# so `docker logs ${CONTAINER_NAME}` stays available for post-mortem.
