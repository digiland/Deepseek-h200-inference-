#!/bin/bash
# Start both copies on one node, one after the other (two at once stall on kernel compiles).
# usage: start_node.sh <best|tput>
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); . "$D/node.env"; mkdir -p "$V41_LOGS"; P=${1:?usage: start_node.sh <best|tput>}
up() { for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$1/health" >/dev/null && return 0; kill -0 "$2" 2>/dev/null || { echo "copy on port $1 exited, see $V41_LOGS/serve-$1.log"; return 1; }; sleep 15; done; echo "copy on port $1 not healthy after 60 min"; return 1; }
setsid bash "$D/serve_copy.sh" "$P" "$V41_PORT_A" 0,1,2,3 & A=$!; echo "$A" > "$V41_LOGS/copy-a.pid"
up "$V41_PORT_A" "$A" && echo "copy A healthy on port $V41_PORT_A"
setsid bash "$D/serve_copy.sh" "$P" "$V41_PORT_B" 4,5,6,7 & B=$!; echo "$B" > "$V41_LOGS/copy-b.pid"
up "$V41_PORT_B" "$B" && echo "copy B healthy on port $V41_PORT_B"
