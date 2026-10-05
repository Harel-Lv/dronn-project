#!/usr/bin/env python3
"""Launch dronn_rt_service using rt_control settings from config.yaml."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SERVICE = _ROOT / "rt_control" / "build" / "dronn_rt_service"


def main() -> None:
    if os.name == "nt":
        print("Use WSL/Linux: bash scripts/build_rt_control.sh && python scripts/start_rt_service.py")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from app.config import load_config

    cfg = load_config(args.config)
    rt = cfg.get("rt_control", {})
    port = int(rt.get("port", 9999))
    hz = float(rt.get("control_hz", rt.get("loop_hz", 20)))
    bind = str(rt.get("service_bind", "0.0.0.0"))
    hover = int(rt.get("watchdog_hover_ms", 500))
    safe = int(rt.get("watchdog_safe_ms", 2000))
    stale = int(rt.get("stale_packet_ms", 2000))

    if not SERVICE.is_file():
        print(f"Missing {SERVICE}\nRun: bash scripts/build_rt_control.sh")
        sys.exit(1)

    cmd = [
        str(SERVICE),
        "--bind",
        bind,
        "--port",
        str(port),
        "--hz",
        str(hz),
        "--watchdog-hover",
        str(hover),
        "--watchdog-safe",
        str(safe),
        "--stale-ms",
        str(stale),
    ]
    print("Starting:", " ".join(cmd))
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
