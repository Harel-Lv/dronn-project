#!/usr/bin/env python3
"""
Send a short UDP command sequence to dronn_rt_service (Linux/WSL).

Usage:
  ./scripts/build_rt_control.sh
  python scripts/rt_cpp_ipc_demo.py              # starts service on port 9999
  python scripts/rt_cpp_ipc_demo.py --no-start   # service already running
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SERVICE = _ROOT / "rt_control" / "build" / "dronn_rt_service"


def main() -> None:
    if os.name == "nt":
        print(
            "Skip: dronn_rt_service requires Linux/WSL (POSIX UDP).\n"
            "On WSL: bash scripts/build_rt_control.sh && python scripts/rt_cpp_ipc_demo.py"
        )
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()

    if not SERVICE.is_file() and not args.no_start:
        print(f"Build first: bash scripts/build_rt_control.sh\nMissing: {SERVICE}")
        sys.exit(1)

    proc: subprocess.Popen | None = None
    if not args.no_start:
        proc = subprocess.Popen(
            [str(SERVICE), "--port", str(args.port), "--hz", "20"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.4)
        if proc.poll() is not None:
            print("Service failed to start (port in use or missing binary?)")
            sys.exit(1)

    from app.config import load_config
    from app.drone.rt_cpp_client import RtCppClient

    cfg = load_config()
    cfg.setdefault("rt_control", {})
    cfg["rt_control"]["host"] = "127.0.0.1"
    cfg["rt_control"]["port"] = args.port

    client = RtCppClient(cfg)
    try:
        client.connect()
        client.takeoff()
        for _ in range(5):
            client.send_rc(0, 30, 0, 0)
            time.sleep(0.05)
        client.send_rc(0, 0, 0, 0)
        client.land()
        client.close()
    except OSError as exc:
        print(f"UDP demo failed: {exc}")
        if proc:
            proc.terminate()
        sys.exit(1)

    time.sleep(0.3)
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("OK: rt_cpp IPC demo (connect → takeoff → RC → land → disconnect)")


if __name__ == "__main__":
    main()
