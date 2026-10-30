# Validating `rt_control`

Requires **Linux or WSL** (POSIX UDP). Native Windows builds are not supported for the service.

## Build and unit tests

```bash
bash scripts/build_rt_control.sh
cd rt_control/build && ctest --output-on-failure
```

Expected: `dronn_rt_tests: OK`.

Unit tests cover FSM transitions, safety latch, command queue priority/clear, watchdog timing, packet decode, and timestamp guards.

## Python checks (any OS)

```bash
python scripts/smoke_test.py
```

## Optional UDP integration (Linux/WSL, simulator backend)

With the service binary built:

```bash
python scripts/rt_integration_test.py
```

Exercises ACK-matched **connect → takeoff → RC → land → disconnect** and duplicate-sequence rejection.

## Benchmark (Linux/WSL)

```bash
bash scripts/benchmark_rt_loop.sh 15
```

On shutdown the service prints a `[metrics]` summary (measured Hz, avg/max/p95 exec ms, jitter, deadline misses, RC failures).  
Numbers are **environment-specific** — copy the actual line from your run into release notes; do not reuse stale values.

## CI

GitHub Actions workflow `.github/workflows/ci.yml` runs CMake build, CTest, `python -m compileall`, and `scripts/smoke_test.py` on Ubuntu.

## Limitations

- Simulator backend only; not validated on DJI hardware or hard real-time kernels.
- UDP integration tests do not simulate packet loss without additional tooling.
- Benchmark script does not apply synthetic CPU load by default.
