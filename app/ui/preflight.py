"""Lightweight preflight checks for the launcher UI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import time

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"
_WEBCAM_PROBE_CACHE: tuple[int, float, bool] | None = None
_TELLO_PROBE_CACHE: tuple[float, "TelloProbeResult"] | None = None
_TELLO_CACHE_OK_SEC = 12.0
_TELLO_CACHE_FAIL_SEC = 1.5


@dataclass(frozen=True)
class CheckResult:
    label: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class TelloProbeResult:
    connected: bool
    battery: int | None
    detail: str


def _hand_model_path() -> Path:
    return _MODELS_DIR / "hand_landmarker.task"


def _face_model_path() -> Path | None:
    primary = _MODELS_DIR / "face_landmarker.task"
    if primary.is_file():
        return primary
    found = sorted(_MODELS_DIR.glob("face_landmarker*.task"))
    return found[0] if found else None


def _pose_model_path() -> Path:
    return _MODELS_DIR / "pose_landmarker_lite.task"


def _insightface_available() -> bool:
    try:
        import insightface  # noqa: F401

        return True
    except ImportError:
        return False


def _battery_thresholds(config: dict | None) -> tuple[int, int]:
    dcfg = (config or {}).get("drone", {}) or {}
    warn = max(1, int(dcfg.get("battery_warn_percent", 30)))
    block = max(0, int(dcfg.get("battery_block_percent", 15)))
    if block > warn:
        block = warn
    return warn, block


def _tello_sdk_responds(host: str = "192.168.10.1", *, timeout: float = 5.0) -> bool | None:
    """
    Quick UDP SDK check. Tello replies to the source port — bind local 8889 like djitellopy.
    Returns None if port 8889 is already in use (caller should use djitellopy directly).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            sock.bind(("", 8889))
        except OSError:
            return None
        sock.settimeout(timeout)
        sock.sendto(b"command", (host, 8889))
        data, _ = sock.recvfrom(128)
        text = data.decode("utf-8", errors="ignore").strip().lower()
        return text == "ok" or text.endswith("ok")
    except OSError:
        return False
    finally:
        sock.close()


def invalidate_tello_probe_cache() -> None:
    global _TELLO_PROBE_CACHE
    _TELLO_PROBE_CACHE = None


def probe_tello(*, config: dict | None = None, live: bool = True) -> TelloProbeResult:
    """Try a short Tello connect + battery read (no stream, no takeoff)."""
    if not live:
        return TelloProbeResult(
            connected=False,
            battery=None,
            detail="Tello check skipped (simulation mode)",
        )

    global _TELLO_PROBE_CACHE
    now = time.monotonic()
    cached = _TELLO_PROBE_CACHE
    if cached is not None:
        age = now - cached[0]
        ttl = _TELLO_CACHE_OK_SEC if cached[1].connected else _TELLO_CACHE_FAIL_SEC
        if age < ttl:
            return cached[1]

    dcfg = (config or {}).get("drone", {}) or {}
    host_raw = dcfg.get("tello_host")
    host = (
        str(host_raw).strip()
        if host_raw is not None and str(host_raw).strip()
        else "192.168.10.1"
    )

    # Optional quick UDP check; failure still falls through to djitellopy.
    _tello_sdk_responds(host)

    try:
        from djitellopy import Tello
    except ImportError:
        result = TelloProbeResult(
            connected=False,
            battery=None,
            detail="pip install djitellopy av",
        )
        _TELLO_PROBE_CACHE = (now, result)
        return result

    tello = None
    try:
        from app.drone.tello_connect import connect_tello_and_read_battery

        tello = Tello(host=host) if host != "192.168.10.1" else Tello()
        tello.RESPONSE_TIMEOUT = max(5, int(float(dcfg.get("tello_response_timeout_sec", 8))))
        battery = connect_tello_and_read_battery(tello, dcfg)
        result = TelloProbeResult(
            connected=True,
            battery=battery,
            detail=f"connected — battery {battery}%",
        )
    except Exception as exc:
        result = TelloProbeResult(
            connected=False,
            battery=None,
            detail=f"NOT connected — check the drone is on ({exc})",
        )
    finally:
        if tello is not None:
            try:
                tello.end()
            except Exception:
                pass
            # Tello.__del__ calls end() again; stale flags make it hit a removed host (KeyError).
            tello.stream_on = False
            tello.is_flying = False

    _TELLO_PROBE_CACHE = (now, result)
    return result


def _battery_check(
    battery: int | None,
    *,
    warn_pct: int,
    block_pct: int,
) -> CheckResult:
    if battery is None:
        return CheckResult(
            "Tello battery",
            False,
            "not read — check connection",
        )
    b = max(0, min(100, int(battery)))
    if b < block_pct:
        return CheckResult(
            "Tello battery",
            False,
            f"{b}% — too low to fly (minimum {block_pct}%)",
        )
    if b < warn_pct:
        return CheckResult(
            "Tello battery",
            True,
            f"WARNING: {b}% — charge before flying (recommended above {warn_pct}%)",
        )
    return CheckResult("Tello battery", True, f"{b}% — OK")


def probe_webcam(camera_index: int) -> bool:
    global _WEBCAM_PROBE_CACHE
    try:
        from app.camera.webcam import Webcam

        now = time.monotonic()
        cached = _WEBCAM_PROBE_CACHE
        if (
            cached is not None
            and cached[0] == camera_index
            and (now - cached[1]) < 8.0
        ):
            return cached[2]

        cam = Webcam(camera_index=camera_index)
        cam.open()
        frame = cam.read_frame()
        cam.release()
        ok = frame is not None
        _WEBCAM_PROBE_CACHE = (camera_index, now, ok)
        return ok
    except Exception:
        return False


def run_preflight(
    *,
    mode: str,
    live_tello: bool,
    camera_index: int = 0,
    config: dict | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    hand_ok = _hand_model_path().is_file()
    needs_hand = mode in ("gesture", "webcam_faces")
    if needs_hand:
        results.append(
            CheckResult(
                "Model: hand (hand_landmarker)",
                hand_ok,
                "models/hand_landmarker.task" if hand_ok else "missing — see models/README.md",
            )
        )

    needs_face = mode in ("identity", "webcam_faces")
    if needs_face:
        if _insightface_available():
            results.append(CheckResult("Model: face recognition (InsightFace)", True, "installed"))
        else:
            face_path = _face_model_path()
            face_ok = face_path is not None
            results.append(
                CheckResult(
                    "Model: face / InsightFace",
                    face_ok,
                    str(face_path.name) if face_ok else "pip install insightface",
                )
            )

    if mode == "tracking":
        pose_ok = _pose_model_path().is_file()
        results.append(
            CheckResult(
                "Model: pose (body tracking)",
                pose_ok,
                "pose_landmarker_lite.task" if pose_ok else "missing",
            )
        )

    if mode == "chase":
        if_ok = _insightface_available()
        results.append(
            CheckResult(
                "Model: face recognition (InsightFace) — chase",
                if_ok,
                "installed" if if_ok else "pip install insightface onnxruntime",
            )
        )

    cam_ok = probe_webcam(camera_index)
    results.append(
        CheckResult(
            f"Webcam (index {camera_index})",
            cam_ok,
            "active" if cam_ok else "failed to open — check connection",
        )
    )

    if live_tello:
        warn_pct, block_pct = _battery_thresholds(config)
        tello = probe_tello(config=config, live=True)
        results.append(
            CheckResult(
                "Tello drone",
                tello.connected,
                tello.detail,
            )
        )
        if tello.connected:
            results.append(
                _battery_check(
                    tello.battery,
                    warn_pct=warn_pct,
                    block_pct=block_pct,
                )
            )
    else:
        results.append(CheckResult("Mode", True, "SIMULATION (no real drone)"))

    return results


def preflight_ready(
    results: list[CheckResult],
    *,
    mode: str,
    live_tello: bool = False,
) -> bool:
    """Hard requirements — webcam warning alone does not block simulation."""
    for r in results:
        if r.label.startswith("Model:"):
            if not r.ok:
                return False
    if live_tello:
        for r in results:
            if r.label in ("Tello drone", "Tello battery") and not r.ok:
                return False
    if mode in ("gesture", "webcam_faces"):
        return _hand_model_path().is_file()
    if mode == "tracking":
        return _pose_model_path().is_file()
    if mode == "chase":
        return _insightface_available()
    if mode == "identity":
        return _insightface_available() or (_face_model_path() is not None)
    return True
