"""Webcam gesture session: enrollment + main control loop."""
from __future__ import annotations

import time

import cv2
import numpy as np

from app.drone.drone_controller import DroneController
from app.gestures.gesture_rules import (
    BACK,
    DOWN,
    FORWARD,
    HOVER,
    LAND,
    LEFT,
    NO_GESTURE,
    RIGHT,
    ROTATE_CCW,
    ROTATE_CW,
    TAKEOFF,
    UP,
    gesture_to_drone_intent,
)
from app.gestures.hand_selection import select_dominant_hand_landmarks
from app.gestures.intent_stabilizer import GestureIntentStabilizer
from app.gestures.motion_analyzer import (
    MotionAnalyzer,
    likely_open_palm_for_calibration,
    palm_span_pixels,
)
from app.identity.face_profile import (
    FaceProfileStore,
    average_embeddings,
    face_landmarks_to_embedding,
)
from app.ui.hud_session import draw_calibration_overlay, draw_gesture_mode_hud


_FPV_GESTURE_WINDOW = "Tello FPV — מצלמת הרחפן"


def _fpv_display_resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


_RC_SPEED = 48
_INTENT_RC: dict[str, tuple[int, int, int, int]] = {
    FORWARD: (0, _RC_SPEED, 0, 0),
    BACK: (0, -_RC_SPEED, 0, 0),
    LEFT: (-_RC_SPEED, 0, 0, 0),
    RIGHT: (_RC_SPEED, 0, 0, 0),
    UP: (0, 0, _RC_SPEED, 0),
    DOWN: (0, 0, -_RC_SPEED, 0),
    ROTATE_CW: (0, 0, 0, _RC_SPEED),
    ROTATE_CCW: (0, 0, 0, -_RC_SPEED),
}


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(np.median(np.array(xs, dtype=np.float64)))


def _session_mode_from_label(mode_label: str) -> str:
    ml = mode_label.lower()
    if "identity" in ml:
        return "identity"
    if "fast" in ml or "pose" in ml:
        return "fast + pose"
    return "gesture"


def _run_open_palm_calibration(
    webcam,
    hand_detector,
    *,
    window_name: str,
    gcfg: dict,
    dominant_hand: str,
    close_window_on_exit: bool = True,
) -> tuple[float | None, dict]:
    seconds = float(gcfg.get("hand_calibration_seconds", 2.5))
    min_open = max(1, int(gcfg.get("hand_calibration_min_open_frames", 8)))
    if seconds <= 0:
        print("[gestures] כיול יד מושבת (hand_calibration_seconds=0)")
        return None, {"status": "skipped", "samples_ok": 0, "min_samples": min_open}

    deadline = time.time() + seconds
    spans: list[float] = []
    print(
        f"[gestures] כיול: הצג כף פתוחה למצלמה ~{seconds:.1f}s "
        f"(לפחות {min_open} פריימים טובים)"
    )

    try:
        while time.time() < deadline:
            frame = webcam.read_frame()
            if frame is None:
                break
            hres = hand_detector.process_frame(frame)
            hand_detector.draw_landmarks(frame, hres)
            pairs = hand_detector.extract_hands_with_handedness(frame, hres)
            chosen = select_dominant_hand_landmarks(pairs, dominant_hand)
            if chosen is not None and likely_open_palm_for_calibration(chosen):
                ps = palm_span_pixels(chosen)
                if ps is not None and ps > 1.0:
                    spans.append(ps)

            left = max(0.0, deadline - time.time())
            draw_calibration_overlay(
                frame,
                seconds_remaining=left,
                total_seconds=seconds,
                samples_ok=len(spans),
                min_samples=min_open,
                dominant_hand=dominant_hand,
            )
            cv2.imshow(window_name, frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                print("[gestures] כיול בוטל — ממשיכים בלי נרמול מרחק")
                return None, {
                    "status": "cancelled",
                    "samples_ok": len(spans),
                    "min_samples": min_open,
                }
    finally:
        # In manual mode we use a dedicated calibration window and should close it.
        # In gesture mode calibration runs on the main session window and must stay open.
        if close_window_on_exit:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    ref = _median(spans) if len(spans) >= min_open else None
    if ref is not None:
        print(
            f"[gestures] כיול יד: reference palm span = {ref:.1f}px ({len(spans)} samples)"
        )
        return ref, {"status": "ok", "samples_ok": len(spans), "min_samples": min_open}
    print(
        f"[gestures] כיול: לא נאספו מספיק פריימים ({len(spans)}/{min_open}) — "
        "ממשיכים בלי נרמול (נסה תאורה, יד מלאה בפריים, מרחק מהמצלמה)"
    )
    return None, {"status": "failed", "samples_ok": len(spans), "min_samples": min_open}


def run_face_enrollment(
    webcam,
    face_detector,
    store: FaceProfileStore,
    *,
    enroll_frames: int = 25,
    display_name: str = "Owner",
    window_name: str = "Enroll",
) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    collected: list[np.ndarray] = []
    print(f"[enroll] Show your face — collecting {enroll_frames} good frames (q/Esc abort)")

    while len(collected) < enroll_frames:
        frame = webcam.read_frame()
        if frame is None:
            break
        res = face_detector.process_frame(frame)
        face_detector.draw_landmarks(frame, res)
        fl = face_detector.first_face_landmarks(res)
        emb = face_landmarks_to_embedding(list(fl)) if fl is not None else None
        if emb is not None:
            collected.append(emb)
        msg = f"Enrolling {display_name}: {len(collected)}/{enroll_frames}"
        cv2.putText(
            frame,
            msg,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if emb is not None else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Enrollment cancelled")

    avg = average_embeddings(collected)
    if avg is None:
        cv2.destroyWindow(window_name)
        raise RuntimeError("Failed to build embedding")
    store.save(avg, display_name, backend="landmarks")
    print(f"[enroll] Saved profile for {display_name!r} (MediaPipe landmarks — פחות מבחין בין אנשים)")
    cv2.destroyWindow(window_name)


def run_face_enrollment_insightface(
    webcam,
    insight,
    store: FaceProfileStore,
    *,
    enroll_frames: int = 25,
    display_name: str = "Owner",
    window_name: str = "Enroll",
) -> None:
    """Collect ArcFace embeddings (InsightFace); מומלץ לזיהוי אדם אמיתי."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    collected: list[np.ndarray] = []
    print(
        f"[enroll] InsightFace (ArcFace) — collecting {enroll_frames} good frames (q/Esc abort)"
    )

    while len(collected) < enroll_frames:
        frame = webcam.read_frame()
        if frame is None:
            break
        emb, bbox = insight.embed(frame)
        if bbox is not None:
            insight.draw_bbox(frame, bbox)
        if emb is not None:
            collected.append(emb)
        msg = f"Enrolling {display_name}: {len(collected)}/{enroll_frames}"
        cv2.putText(
            frame,
            msg,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if emb is not None else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Enrollment cancelled")

    avg = average_embeddings(collected)
    if avg is None:
        cv2.destroyWindow(window_name)
        raise RuntimeError("Failed to build embedding")
    store.save(avg, display_name, backend="insightface")
    print(f"[enroll] Saved ArcFace profile for {display_name!r}")
    cv2.destroyWindow(window_name)


def run_gesture_control_loop(
    webcam,
    hand_detector,
    config: dict,
    *,
    mode_label: str,
    stable_frames_required: int,
    window_name: str,
    pose_detector=None,
    face_detector=None,
    identity_store: FaceProfileStore | None = None,
    match_threshold: float = 0.88,
    insightface_for_identity=None,
) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    gcfg = config.get("gestures", {}) or {}
    dominant = str(gcfg.get("dominant_hand", "any"))
    stable_normal = max(1, int(gcfg.get("stable_frames_normal", stable_frames_required)))
    stable_critical = max(1, int(gcfg.get("stable_frames_critical", 8)))

    palm_ref, calib_meta = _run_open_palm_calibration(
        webcam,
        hand_detector,
        window_name=window_name,
        gcfg=gcfg,
        dominant_hand=dominant,
        close_window_on_exit=False,
    )
    calib_status = str(calib_meta.get("status", "failed"))

    use_pose = pose_detector is not None
    analyzer = MotionAnalyzer(
        use_pose=use_pose,
        reference_palm_span=palm_ref,
        finger_base_ratio=float(gcfg.get("finger_extension_base_ratio", 1.12)),
        thumb_base_ratio=float(gcfg.get("thumb_extension_base_ratio", 1.08)),
        palm_scale_clamp=(
            float(gcfg.get("palm_scale_clamp_low", 0.72)),
            float(gcfg.get("palm_scale_clamp_high", 1.48)),
        ),
    )
    stabilizer = GestureIntentStabilizer(
        stable_frames_normal=stable_normal,
        stable_frames_critical=stable_critical,
        initial_intent=HOVER,
    )
    drone = DroneController(config)
    drone.connect()
    prev_stable = HOVER

    dcfg = config.get("drone", {}) or {}
    show_fpv_window = bool(dcfg.get("show_tello_fpv", False)) and drone.is_live_tello
    fpv_max_w = max(240, int(dcfg.get("follow_pose_max_width", 640)))
    if show_fpv_window:
        cv2.namedWindow(_FPV_GESTURE_WINDOW, cv2.WINDOW_NORMAL)
        print(
            "[gestures] חלון נוסף: מצלמת הרחפן (FPV). "
            "המחוות מזוהות ממצלמת המחשב; הפקודות נשלחות ל-Tello."
        )

    session_mode = _session_mode_from_label(mode_label)
    max_bad = max(1, int(gcfg.get("max_consecutive_bad_frames", 30)))
    land_on_cam_lost = bool(gcfg.get("land_on_camera_lost", False))
    bad_frames = 0

    try:
        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame = webcam.read_frame()
            if frame is None:
                bad_frames += 1
                try:
                    drone.send_rc(0, 0, 0, 0)
                except Exception:
                    pass
                if bad_frames >= max_bad:
                    print(
                        f"[gestures] מצלמה: {bad_frames} פריימים ריקים ברצף — "
                        "עוצרים RC ויוצאים מהלולאה"
                    )
                    if land_on_cam_lost and drone.is_live_tello:
                        try:
                            drone.land()
                        except Exception as exc:
                            print(f"[gestures] land on camera lost: {exc}")
                    break
                continue
            bad_frames = 0

            pose_lm = None
            if pose_detector is not None:
                pres = pose_detector.process_frame(frame)
                pose_detector.draw_landmarks(frame, pres)
                pose_lm = pose_detector.landmarks_to_pixel_dict(frame, pres)

            hres = hand_detector.process_frame(frame)
            hand_detector.draw_landmarks(frame, hres)
            pairs = hand_detector.extract_hands_with_handedness(frame, hres)
            chosen = select_dominant_hand_landmarks(pairs, dominant)
            hands_list = [chosen] if chosen is not None else []
            gesture = analyzer.analyze(hands_list, pose_lm)
            raw_intent = gesture_to_drone_intent(gesture)
            if raw_intent == NO_GESTURE:
                raw_intent = HOVER

            identity_ok = True
            score = 0.0
            iname = ""
            if identity_store is not None:
                if insightface_for_identity is not None:
                    emb, bbox = insightface_for_identity.embed(frame)
                    if bbox is not None:
                        insightface_for_identity.draw_bbox(frame, bbox)
                    identity_ok, score, iname = identity_store.verify(
                        emb, float(match_threshold)
                    )
                elif face_detector is not None:
                    fres = face_detector.process_frame(frame)
                    face_detector.draw_landmarks(frame, fres)
                    fl = face_detector.first_face_landmarks(fres)
                    emb = face_landmarks_to_embedding(list(fl)) if fl is not None else None
                    identity_ok, score, iname = identity_store.verify(
                        emb, float(match_threshold)
                    )
                else:
                    identity_ok = False
                    score = 0.0
                    iname = ""
                if not identity_ok:
                    raw_intent = HOVER

            stable = stabilizer.update(raw_intent)

            if stable == TAKEOFF and prev_stable != TAKEOFF:
                try:
                    drone.takeoff()
                except Exception as exc:
                    print(f"[drone] takeoff error: {exc}")
                    try:
                        drone.send_rc(0, 0, 0, 0)
                    except Exception:
                        pass
            elif stable == LAND and prev_stable != LAND:
                try:
                    drone.send_rc(0, 0, 0, 0)
                    drone.land()
                except Exception as exc:
                    print(f"[drone] land error: {exc}")
                    try:
                        drone.send_rc(0, 0, 0, 0)
                    except Exception:
                        pass
            elif stable in _INTENT_RC:
                drone.send_rc(*_INTENT_RC[stable])
            else:
                drone.send_rc(0, 0, 0, 0)

            prev_stable = stable

            identity_line = None
            if identity_store is not None:
                identity_line = (
                    f"identity gate: ok={identity_ok}  score={score:.2f}  ({iname})"
                )

            draw_gesture_mode_hud(
                frame,
                session_mode=session_mode,
                dominant_hand=dominant,
                calib_status=calib_status,
                calib_ref_px=palm_ref,
                raw_gesture=gesture,
                raw_intent=raw_intent,
                stable_intent=stable,
                identity_line=identity_line,
                stab_normal=stable_normal,
                stab_critical=stable_critical,
                is_live_drone=drone.is_live_tello,
            )

            if show_fpv_window:
                fpv = drone.read_fpv_frame()
                if fpv is not None:
                    disp = _fpv_display_resize(fpv, fpv_max_w)
                    cv2.putText(
                        disp,
                        "Tello FPV",
                        (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(_FPV_GESTURE_WINDOW, disp)

            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
    finally:
        if show_fpv_window:
            try:
                cv2.destroyWindow(_FPV_GESTURE_WINDOW)
            except cv2.error:
                pass
        try:
            drone.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        drone.disconnect()
