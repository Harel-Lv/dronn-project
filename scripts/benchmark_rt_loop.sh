#!/usr/bin/env bash
# Measure dronn_rt_service control-loop timing (Linux/WSL). Requires a built binary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/rt_control/build"
BIN="${BUILD}/dronn_rt_service"
DURATION="${1:-10}"

if [[ ! -x "$BIN" ]]; then
  echo "Build first: bash scripts/build_rt_control.sh" >&2
  exit 1
fi

PORT=19999
export DRONN_RT_PORT="$PORT"
"$BIN" --port "$PORT" --hz 20 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT
sleep 0.5

python3 - <<'PY'
import socket, struct, time
host, port = "127.0.0.1", int(__import__("os").environ.get("DRONN_RT_PORT", "19999"))
fmt = "<BBH I Q hhhh"
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(struct.pack(fmt, 1, 13, 0, 1, int(time.time()*1e6), 0,0,0,0), (host, port))
ack = s.recvfrom(8)[0]
assert ack[0] == 0xAC
PY

python3 - <<PY
import socket, struct, time
host, port = "127.0.0.1", $PORT
fmt = "<BBH I Q hhhh"
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(2.0)
def send(cmd, seq):
    s.sendto(struct.pack(fmt, 1, cmd, 0, seq, int(time.time()*1e6), 0,0,0,0), (host, port))
    return s.recvfrom(8)[0]
send(13, 1)
send(1, 2)
PY

echo "Collecting metrics for ${DURATION}s (send periodic RC to reset watchdog)..."
end=$(($(date +%s) + DURATION))
seq=3
while [[ $(date +%s) -lt $end ]]; do
  python3 - <<PY
import socket, struct, time
host, port = "127.0.0.1", $PORT
fmt = "<BBH I Q hhhh"
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(struct.pack(fmt, 1, 12, 0, $seq, int(time.time()*1e6), 0,0,0,0), (host, port))
PY
  seq=$((seq + 1))
  sleep 0.2
done

kill -INT "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
trap - EXIT

echo "--- Last metrics line from service stderr (if printed on shutdown) ---"
echo "Re-run with: DRONN_RT_LOG=1 or inspect service metrics() output in code."
echo "For detailed exec stats, enable periodic metrics logging in main.cpp if not present."
