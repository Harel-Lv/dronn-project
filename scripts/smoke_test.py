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


class _FakeTello:
    """Just enough of djitellopy.Tello for DroneController logic (no network)."""

    def __init__(self, *, h=80, bat=60, temph=60, takeoff_error=False, land_error=False):
        self.state = {"h": h, "bat": bat, "temph": temph}
        self.retry_count = 3
        self.takeoff_error = takeoff_error
        self.land_error = land_error
        self.rc: list[tuple] = []
        self.lands = 0

    def get_current_state(self):
        return self.state

    def get_battery(self):
        return self.state["bat"]

    def send_rc_control(self, *a):
        self.rc.append(a)

    def send_command_with_return(self, *_a, **_k):
        return "error"

    def takeoff(self):
        if self.takeoff_error:
            raise Exception("error")

    def land(self):
        self.lands += 1
        if self.land_error:
            raise Exception("error")


def _live_drone(**tello_kw):
    from app.config import load_config
    from app.drone.drone_controller import DroneController

    cfg = load_config()
    cfg["drone"].update(enabled=True, backend="tello")
    d = DroneController(cfg)
    d._tello = _FakeTello(**tello_kw)
    return d


def _drone_logic_checks() -> None:
    import time

    import numpy as np

    print("smoke: drone logic (fake Tello)")
    d = _live_drone(h=80)
    d.land()
    if d._tello.lands != 1 or d.is_airborne:
        _fail("L must land a Tello that reports height even when the flag says grounded")
    d = _live_drone(h=0)
    d.land()
    if d._tello.lands != 0:
        _fail("L on the ground (h=0) must not send land")
    _ok("land always reaches a flying Tello")

    d = _live_drone(h=90, takeoff_error=True)
    d.takeoff()
    if not d.is_airborne:
        _fail("lost takeoff reply with height>0 must mark airborne")
    d = _live_drone(h=0, temph=95, takeoff_error=True)
    try:
        d.takeoff()
        _fail("hot takeoff should raise")
    except RuntimeError as exc:
        if "too hot" not in str(exc):
            _fail(f"hot takeoff message: {exc}")
    _ok("takeoff resync + refusal reason")

    class Reader:
        frame = np.zeros([300, 400, 3], dtype=np.uint8)

    d = _live_drone()
    d._frame_reader = Reader()
    if d.read_fpv_frame() is not None:
        _fail("djitellopy placeholder must read as no frame")
    rgb = np.zeros((720, 960, 3), np.uint8)
    rgb[..., 0] = 200
    Reader.frame = rgb
    out = d.read_fpv_frame()
    if out is None or out[0, 0, 2] != 200 or out[0, 0, 0] != 0:
        _fail("FPV must convert RGB to BGR")
    out[:] = 255
    if d.read_fpv_frame()[0, 0, 2] != 200:
        _fail("FPV frames must be copies (HUD drawing leaked into the cache)")
    d._fpv_last_new_at -= 5.0
    if d.read_fpv_frame() is not None:
        _fail("frozen FPV stream must read as no frame")
    _ok("FPV placeholder / BGR / copy / freeze")

    d = _live_drone()
    d._cfg = dict(d._cfg, keepalive_idle_sec=1.0)
    d._airborne = True
    d._last_cmd_at = time.monotonic()
    d._start_keepalive()
    time.sleep(1.7)
    d._stop_keepalive()
    if (0, 0, 0, 0) not in d._tello.rc:
        _fail("keepalive must send hover RC when idle while airborne")
    _ok("keepalive hover RC")

    d = _live_drone(bat=50)
    d._cfg = dict(d._cfg, link_lost_seconds=0.5)
    d._airborne = True
    if d.poll_flight_status().link_lost:
        _fail("fresh telemetry is not link lost")
    time.sleep(0.6)
    if not d.poll_flight_status().link_lost:
        _fail("no new telemetry packets must report LINK LOST")
    d._tello.state = {"h": 80, "bat": 9, "temph": 60}
    d._battery_display = None
    d._last_battery_at = 0.0
    d.poll_flight_status()
    if d._tello.lands != 1 or d.is_airborne:
        _fail("critical battery must auto-land")
    _ok("flight status: link lost + battery auto-land")

    from app.session_modes import is_quit_key

    d = _live_drone()
    d._airborne = True
    if is_quit_key(ord("q"), d) or not is_quit_key(27, d):
        _fail("q must not quit while airborne; Esc always quits")
    _ok("quit keys")


def _control_checks() -> None:
    from types import SimpleNamespace

    import numpy as np

    print("smoke: follow / chase control")
    from app.drone.rc_smoothing import RcSmoother
    from app.run_body_follow import BodyMetrics, _compute_follow_rc, _limit_height

    m = BodyMetrics(cx=900, cy=600, size_ratio=0.1, bbox=(0, 0, 1, 1))
    lr, _fb, ud, yaw = _compute_follow_rc(
        m, 960, center_deadzone=0.1, size_min_ratio=0.04, size_max_ratio=0.2,
        max_speed=48, frame_h=720, vertical_gain=40.0,
    )
    if not (lr > 0 and yaw > 0):
        _fail("target right of center must strafe and yaw right")
    if not ud < 0:
        _fail("target low in frame must descend")
    if _limit_height(-20, 50, 60, 220) != 0 or _limit_height(20, 230, 60, 220) != 0:
        _fail("height band must block descend below min / climb above max")
    _ok("follow direction + height band")

    s = RcSmoother(alpha=0.5, max_step=10)
    first = s.update((100, 0, 0, 0))
    if first[0] > 10:
        _fail("smoother must rate-limit")
    if s.update((0, 0, 0, 0)) != (0, 0, 0, 0):
        _fail("zero target (stop) must bypass smoothing")
    _ok("RC smoother")

    print("smoke: identity")
    import os
    import tempfile

    from app.identity.face_profile import FaceProfileStore, cosine_match, face_landmarks_to_embedding

    rng = np.random.default_rng(0)
    shape = rng.uniform(-0.5, 0.5, (478, 2))

    def place(cx, cy, scale):
        return [SimpleNamespace(x=cx + scale * x, y=cy + scale * y, z=0.0) for x, y in shape]

    a = face_landmarks_to_embedding(place(0.5, 0.5, 0.2))
    b = face_landmarks_to_embedding(place(0.2, 0.75, 0.08))
    if cosine_match(a, b) < 0.999:
        _fail("landmark embedding must not depend on face position / distance")
    _ok("landmark embedding invariance")

    store = FaceProfileStore(os.path.join(tempfile.mkdtemp(), "p.npz"), "Owner")
    store.save(a, "Owner", backend="landmarks")
    store.verify(a, 0.9)
    loads = {"n": 0}
    real_load = np.load

    def counting_load(*args, **kw):
        loads["n"] += 1
        return real_load(*args, **kw)

    np.load = counting_load
    try:
        for _ in range(20):
            store.verify(a, 0.9)
    finally:
        np.load = real_load
    if loads["n"] != 0:
        _fail("profile must be cached between verifies")
    _ok("profile cache")


def _synthetic_hand(fingers: str, thumb: str = "in") -> dict[int, tuple[int, int]]:
    """Upright hand (wrist at origin, fingers toward -y). fingers: 4 chars of 1/0 = index..pinky."""
    lm: dict[int, tuple[int, int]] = {0: (0, 0), 9: (-10, -80), 5: (-30, -80)}
    for (tip, pip), x, up in zip(((8, 6), (12, 10), (16, 14), (20, 18)), (-30, -10, 10, 30), fingers):
        lm[pip] = (x, -100)
        lm[tip] = (x, -160 if up == "1" else -70)
    lm[3] = (-60, -50)
    lm[4] = {"in": (-35, -60), "relaxed": (-55, -75), "out": (-100, -60)}[thumb]
    return lm


def _gesture_checks() -> None:
    from app.gestures.gesture_rules import gesture_to_drone_intent
    from app.gestures.motion_analyzer import MotionAnalyzer

    print("smoke: hand gestures")
    an = MotionAnalyzer()
    cases = {
        ("0000", "in"): ("FIST", "LAND"),
        ("1111", "out"): ("OPEN_PALM", "HOVER"),
        ("0000", "out"): ("THUMB_UP", "TAKEOFF"),
        ("1000", "in"): ("POINT", "FORWARD"),
        ("1000", "relaxed"): ("POINT", "FORWARD"),
        ("1100", "in"): ("PEACE", "BACK"),
        ("0001", "in"): ("PINKY", "LEFT"),
        ("1110", "in"): ("THREE", "RIGHT"),
        ("1001", "in"): ("ROCK", "ROTATE_CCW"),
        ("1000", "out"): ("GUN", "ROTATE_CW"),
    }
    for (fingers, thumb), (want_g, want_i) in cases.items():
        g = an.analyze_hands([_synthetic_hand(fingers, thumb)])
        if g != want_g or gesture_to_drone_intent(g) != want_i:
            _fail(f"fingers={fingers} thumb={thumb}: got {g}/{gesture_to_drone_intent(g)}, want {want_g}/{want_i}")
    _ok(f"{len(cases)} hand signs classified + mapped")


def _session_back_check() -> None:
    import app.session as s
    from app.session import SessionOptions

    print("smoke: session mode fallback")
    results = iter(["gesture", "webcam_faces", s.BACK_TO_PREVIOUS, None])
    visited: list[str] = []

    class Ctx:
        def __init__(self, **_k):
            self.preserve_main_window = False
            self.drone = None

        def prepare_mode_switch(self):
            pass

        def show_mode_switch_overlay(self, _m):
            pass

        def shutdown(self):
            pass

    saved = (s._run_single_mode, s.SessionContext, s.ensure_opencv_gui_windows)
    s._run_single_mode = lambda mode, *_a: (visited.append(mode), next(results))[1]
    s.SessionContext = Ctx
    s.ensure_opencv_gui_windows = lambda: None
    try:
        s.run_session(SessionOptions(mode="manual"))
    finally:
        s._run_single_mode, s.SessionContext, s.ensure_opencv_gui_windows = saved
    if visited != ["manual", "gesture", "webcam_faces", "gesture"]:
        _fail(f"mode that can't load must return to the previous mode: {visited}")
    _ok("back to previous mode")


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
    if not warn.ok or "WARNING" not in warn.detail:
        _fail("battery warn threshold")
    if not _battery_check(55, warn_pct=30, block_pct=15).ok:
        _fail("battery ok threshold")
    _ok("battery preflight rules")
    probe_tello(config={"drone": {}}, live=False)
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

    _drone_logic_checks()
    _control_checks()
    _gesture_checks()
    _session_back_check()

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
