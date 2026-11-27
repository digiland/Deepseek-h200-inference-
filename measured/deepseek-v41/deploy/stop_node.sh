#!/bin/bash
# Stop both copies and wait until GPU memory is free.
D=$(cd "$(dirname "$0")" && pwd); . "$D/node.env"
for p in "$V41_PORT_A" "$V41_PORT_B"; do pkill -f "api_server.*--port $p" 2>/dev/null; done
for f in "$V41_LOGS"/copy-*.pid; do [ -f "$f" ] && kill -- -"$(cat "$f")" 2>/dev/null; rm -f "$f"; done
for _ in $(seq 1 40); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 4000 ] && { echo "GPUs free"; exit 0; }; sleep 5; done; echo "GPUs still busy"
