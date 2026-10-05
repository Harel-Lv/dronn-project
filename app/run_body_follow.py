"""Body-follow mode: Tello FPV + MediaPipe Pose — lock on person at entry, send RC."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.drone.drone_controller import DroneController
from app.identity.face_profile import FaceProfileStore
from app.identity.identity_helpers import resolve_identity_detectors
from app.identity.identity_perf import CachedInsightFaceVerifier, identity_perf_settings
from app.session_modes import MODE_SWITCH_HINT, MODE_SWITCH_KEYS, SESSION_MAIN_WINDOW
from app.ui.operator_hud import draw_operator_hud


_FOLLOW_WINDOW = "Body follow — Tello FPV"
_SIM_WINDOW = SESSION_MAIN_WINDOW


@dataclass
class BodyMetrics:
    cx: int
    cy: int
    size_ratio: float
    bbox: tuple[int, int, int, int]


def _resize_for_pose(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, 1.0 / scale


def _metrics_on_frame(
    pose_detector,
    frame: np.ndarray,
    max_width: int,
) -> BodyMetrics | None:
    small, inv_scale = _resize_for_pose(frame, max_width)
    res = pose_detector.process_frame(small)
    lm = pose_detector.landmarks_to_pixel_dict(small, res)
    if not lm:
        return None

    fh, fw = frame.shape[:2]
    if 23 in lm and 24 in lm:
        cx = (lm[23][0] + lm[24][0]) / 2.0
        cy = (lm[23][1] + lm[24][1]) / 2.0
    elif 11 in lm and 12 in lm:
        cx = (lm[11][0] + lm[12][0]) / 2.0
        cy = (lm[11][1] + lm[12][1]) / 2.0
    elif 0 in lm:
        cx, cy = lm[0]
    else:
        return None

    xs = [p[0] * inv_scale for p in lm.values()]
    ys = [p[1] * inv_scale for p in lm.values()]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    return BodyMetrics(
        cx=int(cx * inv_scale),
        cy=int(cy * inv_scale),
        size_ratio=(bw * bh) / max(1, fw * fh),
        bbox=(x0, y0, x1, y1),
    )


def _compute_follow_rc(
    metrics: BodyMetrics,
    frame_w: int,
    *,
    center_deadzone: float,
    size_min_ratio: float,
    size_max_ratio: float,
    max_speed: int,
) -> tuple[int, int, int, int]:
    err_x = (metrics.cx - frame_w * 0.5) / max(frame_w * 0.5, 1.0)
    lr = yaw = fb = 0
    if abs(err_x) > center_deadzone:
        lr = int(np.clip(-err_x * 62.0, -max_speed, max_speed))
        yaw = int(np.clip(err_x * 48.0, -max_speed, max_speed))
    if metrics.size_ratio < size_min_ratio:
        fb = int(max_speed * 0.55)
    elif metrics.size_ratio > size_max_ratio:
        fb = int(-max_speed * 0.45)
    return lr, fb, 0, yaw


def _draw_follow_hud(
    frame: np.ndarray,
    *,
    metrics: BodyMetrics | None,
    locked: bool,
    status: str,
    lr: int,
    fb: int,
    yaw: int,
    is_sim: bool,
) -> None:
    h, w = frame.shape[:2]
    if metrics is not None:
        x0, y0, x1, y1 = metrics.bbox
        color = (0, 220, 0) if locked else (0, 200, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.circle(frame, (metrics.cx, metrics.cy), 6, color, -1)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (80, 80, 120), 1)
    calibrating = str(status).upper().startswith("CALIB")
    draw_operator_hud(
        frame,
        mode="BODY FOLLOW",
        command=status,
        detail=None if calibrating else f"RC {lr},{fb},{yaw}",
        is_sim=is_sim,
        mode_for_keys="tracking",
        calibrating=calibrating,
        command_locked=locked if locked and not calibrating else None,
    )


def run_body_follow_loop(
    config: dict,
    *,
    webcam,
    pose_detector,
    session_ctx=None,
) -> str | None:
    """
    Follow the person locked during calibration using body pose on FPV (or webcam in sim).
  """
    tcfg = config.get("tracking", {}) or {}
    dcfg = config.get("drone", {}) or {}
    icfg = config.get("identity", {}) or {}

    process_every = max(1, int(tcfg.get("process_every_n_frames", 2)))
    max_width = max(160, int(tcfg.get("max_width", 480)))
    max_speed = max(20, int(tcfg.get("max_rc_speed", 48)))
    center_deadzone = float(tcfg.get("center_deadzone", 0.12))
    calib_sec = float(tcfg.get("calibration_seconds", 6.0))
    calib_min = max(3, int(tcfg.get("calibration_min_samples", 12)))
    small_factor = float(tcfg.get("calibration_small_factor", 0.78))
    large_factor = float(tcfg.get("calibration_large_factor", 1.32))
    require_identity = bool(tcfg.get("require_identity_profile", False))
    stale_sec = float(dcfg.get("fpv_stale_seconds", 5.0))

    owns_drone = session_ctx is None
    if session_ctx is not None:
        drone = session_ctx.ensure_drone(config)
    else:
        drone = DroneController(config)
        drone.connect()
    use_fpv = bool(dcfg.get("show_tello_fpv", False)) and drone.is_live_tello
    win = _FOLLOW_WINDOW if use_fpv else _SIM_WINDOW
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    identity_verifier: CachedInsightFaceVerifier | None = None
    insight_backend = None
    face_lm_cleanup = None
    store: FaceProfileStore | None = None
    if require_identity or icfg.get("profile_path"):
        store = FaceProfileStore(
            str(icfg.get("profile_path", "data/identity/profile.npz")),
            str(icfg.get("display_name", "Owner")),
        )
    if require_identity and store is not None and store.exists():
        eb, threshold, face_lm_cleanup, insight = resolve_identity_detectors(config)
        insight_backend = insight
        if eb == "insightface" and insight is not None:
            perf_every, perf_max_w = identity_perf_settings(config)
            identity_verifier = CachedInsightFaceVerifier(
                insight,
                store,
                match_threshold=float(threshold),
                process_every_n_frames=perf_every,
                max_width=perf_max_w,
            )
        elif require_identity:
            print(
                "[follow] require_identity_profile=true אבל אין פרופיל/InsightFace — "
                "ממשיכים בנעילת גוף בלבד"
            )

    print(
        f"[follow] מקור וידאו: {'Tello FPV' if use_fpv else 'מצלמת מחשב (סימולציה)'}\n"
        f"  עמוד במרכז הפריים ~{calib_sec:.0f}s לנעילה על הגוף | T המראה | L נחיתה | q יציאה"
    )
    print(f"[session] switch modes: {MODE_SWITCH_HINT}")

    requested_mode: str | None = None

    # --- calibration: lock body size ---
    size_min = float(tcfg.get("size_min_ratio", 0.04))
    size_max = float(tcfg.get("size_max_ratio", 0.22))
    if session_ctx is not None and session_ctx.tracking_calib is not None:
        baseline, size_min, size_max = session_ctx.tracking_calib
        print(
            f"[follow] reusing body lock — baseline={baseline:.3f} "
            f"min={size_min:.3f} max={size_max:.3f}"
        )
    else:
        deadline = time.monotonic() + calib_sec
        size_samples: list[float] = []
        identity_ok_samples = 0

        while time.monotonic() < deadline:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame = _read_follow_frame(drone, webcam, use_fpv)
            if frame is None:
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                continue

            metrics = _metrics_on_frame(pose_detector, frame, max_width)
            if metrics is not None:
                size_samples.append(metrics.size_ratio)
                if identity_verifier is not None:
                    if identity_verifier.process(frame).ok:
                        identity_ok_samples += 1

            rem = max(0.0, deadline - time.monotonic())
            status = f"CALIB {rem:.1f}s samples={len(size_samples)}/{calib_min}"
            _draw_follow_hud(
                frame,
                metrics=metrics,
                locked=False,
                status=status,
                lr=0,
                fb=0,
                yaw=0,
                is_sim=not drone.is_live_tello,
            )
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if MODE_SWITCH_KEYS.get(key):
                requested_mode = MODE_SWITCH_KEYS[key]
                break
            if key in (27, ord("q")):
                break
            if key in (ord("l"), ord("L")):
                try:
                    drone.send_rc(0, 0, 0, 0)
                    if drone.is_airborne:
                        drone.land()
                    print("[follow] land (calib)")
                except Exception as exc:
                    print(f"[follow] land: {exc}")

        if len(size_samples) < calib_min:
            print(
                f"[follow] calib: too few samples ({len(size_samples)}/{calib_min}) — "
                "using defaults"
            )
        else:
            baseline = sum(size_samples) / len(size_samples)
            size_min = max(0.005, baseline * small_factor)
            size_max = min(0.85, baseline * large_factor)
            print(
                f"[follow] body locked — baseline={baseline:.3f} "
                f"min={size_min:.3f} max={size_max:.3f}"
            )
            if session_ctx is not None:
                session_ctx.tracking_calib = (baseline, size_min, size_max)

        if require_identity and identity_verifier is not None:
            if identity_ok_samples < max(1, calib_min // 3):
                print("[follow] warning: weak face ID during calib — body lock only")

        if requested_mode:
            if session_ctx is not None:
                session_ctx.finish_mode_window(win, requested_mode)
            return requested_mode

    locked = True
    frame_idx = 0
    last_good_video = time.monotonic()
    last_metrics: BodyMetrics | None = None
    fpv_stale_warned = False

    try:
        while True:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            raw = _read_follow_frame(drone, webcam, use_fpv)
            if raw is not None:
                frame = raw
                last_good_video = time.monotonic()
                fpv_stale_warned = False
            else:
                frame = _blank_frame("No video frame")

            video_stale = (
                use_fpv
                and stale_sec > 0
                and drone.is_live_tello
                and (time.monotonic() - last_good_video) >= stale_sec
            )
            if video_stale and not fpv_stale_warned:
                print(f"[follow] FPV לא מתעדכן — מאפסים RC (>{stale_sec:.0f}s)")
                fpv_stale_warned = True

            lr = fb = yaw = 0
            status = "FOLLOW"
            metrics: BodyMetrics | None = None

            if not video_stale and frame is not None and frame.size > 0:
                if frame_idx % process_every == 0:
                    metrics = _metrics_on_frame(pose_detector, frame, max_width)
                    if metrics is not None:
                        last_metrics = metrics
                else:
                    metrics = last_metrics

                if metrics is not None:
                    lr, fb, _, yaw = _compute_follow_rc(
                        metrics,
                        frame.shape[1],
                        center_deadzone=center_deadzone,
                        size_min_ratio=size_min,
                        size_max_ratio=size_max,
                        max_speed=max_speed,
                    )
                else:
                    status = "NO BODY — hover"
            else:
                status = "VIDEO STALE — hover"

            try:
                if drone.is_airborne and not video_stale and metrics is not None:
                    drone.send_rc(lr, fb, 0, yaw)
                else:
                    drone.send_rc(0, 0, 0, 0)
            except Exception as exc:
                print(f"[follow] RC: {exc}")

            if not drone.is_airborne and metrics is not None:
                status = "GROUNDED — press T to take off"

            _draw_follow_hud(
                frame,
                metrics=metrics,
                locked=locked,
                status=status,
                lr=lr,
                fb=fb,
                yaw=yaw,
                is_sim=not drone.is_live_tello,
            )
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            requested = MODE_SWITCH_KEYS.get(key)
            if requested:
                requested_mode = requested
                break
            if key in (27, ord("q")):
                break
            if key in (ord("t"), ord("T")):
                try:
                    drone.takeoff()
                    print("[follow] takeoff")
                except Exception as exc:
                    print(f"[follow] takeoff: {exc}")
            if key in (ord("l"), ord("L")):
                try:
                    drone.send_rc(0, 0, 0, 0)
                    drone.land()
                    print("[follow] land")
                except Exception as exc:
                    print(f"[follow] land: {exc}")

            frame_idx += 1
    finally:
        if insight_backend is not None:
            try:
                insight_backend.close()
            except Exception:
                pass
        if face_lm_cleanup is not None:
            try:
                face_lm_cleanup.close()
            except Exception:
                pass
        try:
            drone.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        if owns_drone:
            drone.disconnect()
        if session_ctx is not None:
            session_ctx.finish_mode_window(win, requested_mode)
        else:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
    return requested_mode


def _read_follow_frame(drone: DroneController, webcam, use_fpv: bool) -> np.ndarray | None:
    if use_fpv:
        return drone.read_fpv_frame()
    if webcam is not None:
        return webcam.read_frame()
    return None


def _blank_frame(message: str) -> np.ndarray:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        img,
        message,
        (24, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return img
