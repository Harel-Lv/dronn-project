"""Quick smoke checks — no GUI, no drone hardware."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    raise SystemExit(1)


def main() -> None:
    print("smoke: imports")
    for mod in (
        "app.session",
        "app.ui.preflight",
        "app.ui.launcher",
        "app.drone.drone_controller",
        "main",
    ):
        try:
            importlib.import_module(mod)
            _ok(mod)
        except Exception as exc:
            _fail(f"{mod}: {exc}")

    print("smoke: SessionOptions")
    from app.session import SessionOptions, MODE_CHOICES, apply_options_to_config
    from app.config import load_config

    opts = SessionOptions(mode="gesture", tello=False, fpv=False, camera=0)
    cfg = apply_options_to_config(load_config(), opts)
    if str(cfg.get("drone", {}).get("backend")) not in ("simulated", "tello", "rt_cpp"):
        _fail("config merge")
    if "rt_control" not in cfg or "port" not in cfg["rt_control"]:
        _fail("rt_control defaults")
    _ok("config merge")

    for mode in ("gesture", "identity", "webcam_faces", "tracking", "manual", "chase"):
        if mode not in MODE_CHOICES:
            _fail(f"missing mode {mode}")
    if "fast" in MODE_CHOICES:
        _fail("removed mode fast should not be in MODE_CHOICES")
    _ok("mode choices")

    print("smoke: preflight")
    from app.ui.preflight import preflight_ready, run_preflight

    results = run_preflight(mode="gesture", live_tello=False, camera_index=0)
    if not isinstance(results, list) or not results:
        _fail("preflight empty")
    _ok(f"preflight ({len(results)} checks)")

    ready = preflight_ready(results, mode="gesture", live_tello=False)
    if not ready:
        _fail("gesture preflight should pass when hand model exists")
    _ok("preflight ready")

    from app.ui.preflight import _battery_check, probe_tello

    if _battery_check(10, warn_pct=30, block_pct=15).ok:
        _fail("battery block threshold")
    warn = _battery_check(22, warn_pct=30, block_pct=15)
    if not warn.ok or "אזהרה" not in warn.detail:
        _fail("battery warn threshold")
    if not _battery_check(55, warn_pct=30, block_pct=15).ok:
        _fail("battery ok threshold")
    _ok("battery preflight rules")
    probe_tello(config={"drone": {}})
    _ok("probe_tello callable")

    print("smoke: DroneController safety")
    from app.drone.drone_controller import DroneController

    dc = DroneController(cfg)
    dc.takeoff()
    if not dc.is_airborne:
        _fail("sim takeoff should set airborne")
    dc.land()
    if dc.is_airborne:
        _fail("sim land should clear airborne")
    _ok("airborne flag")

    print("smoke: rt_cpp packet")
    import struct

    from app.drone.rt_cpp_client import RtCppClient

    rt_cfg = load_config()
    client = RtCppClient(rt_cfg)
    client._sock = __import__("socket").socket(__import__("socket").AF_INET, __import__("socket").SOCK_DGRAM)
    client._send("hover", lr=0, fb=0, ud=0, yaw=0)
    if client._sequence != 1:
        _fail("rt sequence")
    _ok("RtCppClient packet encode")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
