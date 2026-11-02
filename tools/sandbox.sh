#!/usr/bin/env bash
# Resource-capped sandbox for Local3D+ jobs (ComfyUI, lowpoly, publish, evaluate) on a shared machine.
#
#   tools/sandbox.sh run   NAME [--mem 12G] [--cpus 8] [--ttl 3h] [--rw DIR]... -- CMD ARGS...
#   tools/sandbox.sh stop  NAME | --all
#   tools/sandbox.sh ls
#   tools/sandbox.sh logs  NAME [-f]
#   tools/sandbox.sh stats NAME          # current / peak memory, CPU time
#
# Each job is a transient systemd user unit (cgroup v2) with:
#   MemoryMax=<mem> (hard: OOM-kill inside the box; no MemoryHigh throttle, which with swap off only stalls)
#   MemorySwapMax=0
#   (swap here is zram = RAM; a job that swaps slows every other agent, so it is not allowed to)
#   CPUQuota=<cpus>*100%   Nice=10   IOWeight=20 (yields IO)   TasksMax=512   RuntimeMaxSec=<ttl>
# and runs inside bubblewrap: the whole filesystem read-only except the project's output dirs, ComfyUI's
# input/output/temp/user dirs and any --rw dirs; private /tmp; GPU devices visible; Hugging Face offline.
set -euo pipefail
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}

ROOT=/bigdata/agent-tom/projects/trellis
COMFY=/bigdata/agent-tom/qwen-image/ComfyUI
LOGDIR=$ROOT/logs/sandbox
PREFIX=l3d
mkdir -p "$LOGDIR" "$ROOT/.cache/sandbox"

unit() { echo "$PREFIX-$1"; }

to_bytes() {  # 12G -> bytes
  local v=${1^^}
  case $v in *G) echo $(( ${v%G} * 1024 * 1024 * 1024 ));; *M) echo $(( ${v%M} * 1024 * 1024 ));; *) echo "$v";; esac
}

to_secs() {  # 3h / 90m / 600 -> seconds
  local v=$1
  case $v in *h) echo $(( ${v%h} * 3600 ));; *m) echo $(( ${v%m} * 60 ));; *s) echo "${v%s}";; *) echo "$v";; esac
}

cmd=${1:-}; shift || true
case $cmd in
run)
  name=$1; shift
  mem=12G; cpus=8; ttl=3h; rw=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case $1 in
      --mem) mem=$2; shift 2;;
      --cpus) cpus=$2; shift 2;;
      --ttl) ttl=$2; shift 2;;
      --rw) rw+=("$2"); shift 2;;
      *) echo "unknown option $1" >&2; exit 2;;
    esac
  done
  [ "${1:-}" = "--" ] && shift
  [ $# -gt 0 ] || { echo "missing command" >&2; exit 2; }
  log=$LOGDIR/$name.log
  binds=()
  for d in "$ROOT/outputs" "$ROOT/logs" "$ROOT/samples" "$ROOT/reports" "$ROOT/test_inputs" "$ROOT/.cache/sandbox" \
           "$COMFY/output" "$COMFY/input" "$COMFY/temp" "$COMFY/user" "${rw[@]}"; do
    mkdir -p "$d"
    binds+=(--bind "$d" "$d")
  done
  systemctl --user reset-failed "$(unit "$name").service" 2>/dev/null || true
  systemd-run --user --quiet --collect --unit="$(unit "$name")" --same-dir \
    -p MemoryMax="$mem" -p MemorySwapMax=0 \
    -p CPUQuota="$(( cpus * 100 ))%" -p Nice=10 -p IOWeight=20 -p TasksMax=512 \
    -p RuntimeMaxSec="$(to_secs "$ttl")" -p KillMode=control-group -p OOMPolicy=kill \
    -p StandardOutput=append:"$log" -p StandardError=append:"$log" \
    --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=LANG="${LANG:-C.UTF-8}" \
    --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
    --setenv=OMP_NUM_THREADS="$cpus" --setenv=MKL_NUM_THREADS="$cpus" \
    --setenv=TRITON_CACHE_DIR="$ROOT/.cache/sandbox/triton" \
    --setenv=TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/sandbox/inductor" \
    --setenv=PYTHONDONTWRITEBYTECODE=1 \
    bwrap --ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp --die-with-parent \
          "${binds[@]}" --chdir "$PWD" -- "$@"
  echo "started $(unit "$name") (mem $mem hard, swap 0, $cpus cpus, ttl $ttl, log $log)"
  ;;
stop)
  if [ "${1:-}" = "--all" ]; then
    systemctl --user list-units --plain --no-legend "$PREFIX-*.service" | awk '{print $1}' | xargs -r systemctl --user stop
  else
    systemctl --user stop "$(unit "$1").service"
  fi
  ;;
ls)
  systemctl --user list-units --plain --no-legend "$PREFIX-*.service" || true
  ;;
logs)
  name=$1; shift
  tail "${@:--n 40}" "$LOGDIR/$name.log"
  ;;
stats)
  systemctl --user show "$(unit "$1").service" -p ActiveState -p MemoryCurrent -p MemoryPeak -p MemorySwapCurrent \
    -p MemoryMax -p CPUUsageNSec -p Result | sed 's/=/: /'
  ;;
*)
  sed -n 2,14p "$0"; exit 2;;
esac
