#!/usr/bin/env bash
# usage: _memmon.sh <pid> <comfy log> <out>
while kill -0 "$1" 2>/dev/null; do
  rss=$(awk '/VmRSS/{printf "%.1f", $2/1048576}' /proc/$1/status 2>/dev/null)
  swp=$(awk '/VmSwap/{printf "%.1f", $2/1048576}' /proc/$1/status 2>/dev/null)
  last=$(tail -c 400 "$2" | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-90)
  echo "$(date +%T) rss=${rss}G swap=${swp}G | $last" >> "$3"
  sleep 1
done
