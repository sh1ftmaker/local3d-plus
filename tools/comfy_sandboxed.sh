#!/usr/bin/env bash
# Start ComfyUI for Local3D+ inside the resource sandbox (see tools/sandbox.sh).
#   tools/comfy_sandboxed.sh [--mem 12G] [--cpus 8] [--ttl 3h]      then: tools/pipeline.py ...
#   tools/sandbox.sh stop comfy
# Memory flags: --cache-none (no node-output cache: dense meshes are GBs each), --disable-pinned-memory (pinned
# staging RAM can't be reclaimed), --reserve-vram 4 (leave VRAM for other GPU users). Add --fast-disk only when
# the weights sit on an SSD: weights then stay disk-backed (reclaimable page cache) instead of anonymous RAM.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mem=12G; cpus=8; ttl=3h; extra=()
while [ $# -gt 0 ]; do
  case $1 in --mem) mem=$2; shift 2;; --cpus) cpus=$2; shift 2;; --ttl) ttl=$2; shift 2;; *) extra+=("$1"); shift;; esac
done
"$here/sandbox.sh" run comfy --mem "$mem" --cpus "$cpus" --ttl "$ttl" -- \
  /bigdata/agent-tom/qwen-image/venv/bin/python /bigdata/agent-tom/qwen-image/ComfyUI/main.py \
  --listen 127.0.0.1 --port 8188 --preview-method none \
  --cache-none --disable-pinned-memory --reserve-vram 4 "${extra[@]}"
for i in $(seq 1 120); do curl -sf localhost:8188/system_stats >/dev/null && { echo "ComfyUI up on :8188"; exit 0; }; sleep 2; done
echo "ComfyUI did not come up; see logs/sandbox/comfy.log" >&2; exit 1
