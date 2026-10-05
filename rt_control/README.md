# dronn_rt_control

**Real-time-oriented periodic control architecture on Linux** (not hard real-time).

Independent C++17 service: fixed-rate control loop, UDP command ingress, flight state machine, watchdog, and a simulator flight backend.

## Build

```bash
bash ../scripts/build_rt_control.sh
# or:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/dronn_rt_tests
```

Requires POSIX UDP (`Linux`, `WSL`, or similar).

## Run

From repo root (reads `config.yaml`):

```bash
python scripts/start_rt_service.py
```

Or manually:

```bash
./build/dronn_rt_service --port 9999 --hz 20 --watchdog-hover 500 --watchdog-safe 2000
```

**Connect ACK:** the service replies with one byte `0xAC` to the client source address after a valid `Connect` packet (Python waits for this).

Python (with `drone.enabled: true`, `drone.backend: rt_cpp`, `rt_control.enabled: true`):

```bash
python ../scripts/rt_cpp_ipc_demo.py
```

## Architecture

- **Thread A:** UDP receiver (blocking I/O, parsing, validation).
- **Thread B:** periodic control loop (`sleep_until`, default 20 Hz).
- **Buffer:** latest-command slot (bounded memory; old commands dropped intentionally).

The control loop never waits on the network. Parsing and stale/sequence checks run in the receiver thread.

## Command packet (v1, 24 bytes, little-endian)

| Field | Type | Notes |
|-------|------|--------|
| version | u8 | `1` |
| command | u8 | see `command.hpp` |
| reserved | u16 | `0` |
| sequence | u32 | strictly increasing |
| timestamp_us | u64 | Python `time.time()` µs |
| lr, fb, ud, yaw | i16 each | clamped to ±100 in C++ |

## Metrics

Every ~5 s the service prints loop period, execution time, jitter, deadline misses, RX/reject counts, watchdog activations, and estimated command-to-control latency (from Python timestamp).

Linux userspace scheduling does **not** guarantee hard real-time deadlines.

## Roadmap (not in this demo slice)

- `TelloUdpBackend` in-process (optional; Python `tello` backend remains)
- ARM64 cross-compile toolchain file
