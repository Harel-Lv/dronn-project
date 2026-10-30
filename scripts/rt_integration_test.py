#!/usr/bin/env python3
"""Optional UDP integration checks against dronn_rt_service (simulator backend).

Run after building and starting the service, e.g.:
  bash scripts/build_rt_control.sh
  ./rt_control/build/dronn_rt_service --port 9999 &
  python scripts/rt_integration_test.py
"""
from __future__ import annotations

import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util
import types

if "app" not in sys.modules:
    sys.modules["app"] = types.ModuleType("app")
if "app.drone" not in sys.modules:
    sys.modules["app.drone"] = types.ModuleType("app.drone")

_safety_path = ROOT / "app" / "drone" / "safety_rc.py"
_safety_spec = importlib.util.spec_from_file_location("app.drone.safety_rc", _safety_path)
assert _safety_spec and _safety_spec.loader
_safety_mod = importlib.util.module_from_spec(_safety_spec)
sys.modules["app.drone.safety_rc"] = _safety_mod
_safety_spec.loader.exec_module(_safety_mod)

_rt_path = ROOT / "app" / "drone" / "rt_cpp_client.py"
_spec = importlib.util.spec_from_file_location("rt_cpp_client", _rt_path)
assert _spec and _spec.loader
_rt_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rt_mod)
RtCppClient = _rt_mod.RtCppClient

_PACKET = struct.Struct("<BBH I Q hhhh")
_ACK = struct.Struct("<BBBB I")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    cfg_path = ROOT / "config.yaml"
    cfg: dict = {"rt_control": {"host": "127.0.0.1", "port": 9999}}
    if cfg_path.is_file():
        try:
            import yaml

            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg = loaded
        except ImportError:
            pass
    rt = dict(cfg.get("rt_control", {}))
    rt["port"] = int(rt.get("integration_port", rt.get("port", 9999)))
    cfg = dict(cfg)
    cfg["rt_control"] = rt

    bin_path = ROOT / "rt_control" / "build" / "dronn_rt_service"
    proc = None
    if bin_path.is_file():
        proc = subprocess.Popen(
            [str(bin_path), "--port", str(rt["port"])],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.4)
    else:
        print("Note: dronn_rt_service binary not found — expecting service already running")

    try:
        client = RtCppClient(cfg)
        client.connect()
        client.takeoff()
        client.send_rc(10, 0, 0, 0)
        client.land()
        client.close()
        print("OK: connect/takeoff/rc/land/disconnect with ACKs")
    except Exception as exc:
        _fail(str(exc))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    # Duplicate sequence must be ignored by server (no crash)
    host = str(rt.get("host", "127.0.0.1"))
    port = int(rt["port"])
    if proc is None and not bin_path.is_file():
        return
    if proc is not None:
        proc = subprocess.Popen(
            [str(bin_path), "--port", str(port)],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2.0)
    ts = int(time.time() * 1_000_000)
    pkt = _PACKET.pack(1, 13, 0, 1, ts, 0, 0, 0, 0)
    s.sendto(pkt, (host, port))
    ack1 = s.recvfrom(8)[0]
    if len(ack1) != 8 or ack1[0] != 0xAC:
        _fail("connect ack shape")
    dup = _PACKET.pack(1, 3, 0, 1, ts, 0, 0, 0, 0)
    s.sendto(dup, (host, port))
    s.settimeout(0.3)
    try:
        s.recvfrom(8)
        _fail("duplicate sequence should not produce a new ack")
    except socket.timeout:
        pass
    s.close()
    if proc is not None:
        proc.terminate()
    print("OK: duplicate command sequence rejected")


if __name__ == "__main__":
    main()
