"""Preview loop: single enrolled face — landmarks or InsightFace ArcFace."""
from __future__ import annotations

import cv2
import numpy as np

from app.detection.face_landmarker_detector import FaceLandmarkerDetector
from app.identity.enroll_ui import run_identity_enroll_wizard
from app.identity.face_profile import FaceProfileStore, face_landmarks_to_embedding
from app.identity.identity_perf import CachedInsightFaceVerifier, identity_perf_settings
from app.ui.operator_hud import draw_flight_status, draw_operator_hud

from app.session_modes import MODE_SWITCH_HINT, MODE_SWITCH_KEYS, is_quit_key


def _landmarks_to_bbox(
    frame: np.ndarray,
    landmarks_list: list,
    *,
    pad_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks_list]
    ys = [lm.y * h for lm in landmarks_list]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = x1 - x0
    dy = y1 - y0
    pad_x = max(dx * pad_ratio, 6.0)
    pad_y = max(dy * pad_ratio, 6.0)
    xi0 = int(max(0, x0 - pad_x))
    yi0 = int(max(0, y0 - pad_y))
    xi1 = int(min(w - 1, x1 + pad_x))
    yi1 = int(min(h - 1, y1 + pad_y))
    return xi0, yi0, xi1, yi1


def _session_drone(session_ctx):
    return session_ctx.drone if session_ctx is not None else None


def _handle_land_key(key: int, session_ctx) -> None:
    """Identity mode sends no RC, but L must still land a drone that was airborne on entry."""
    if key not in (ord("l"), ord("L")) or session_ctx is None:
        return
    drone = session_ctx.drone
    if drone is None:
        return
    print("[identity] EMERGENCY LAND (L)")
    try:
        drone.send_rc(0, 0, 0, 0)
        drone.land()
    except Exception as exc:
        print(f"[identity] land error: {exc}")


def _draw_flight_status(frame: np.ndarray, session_ctx) -> None:
    drone = _session_drone(session_ctx)
    if drone is not None:
        draw_flight_status(frame, drone.poll_flight_status())


def _draw_no_face_overlay(frame: np.ndarray, window_name: str, session_ctx=None) -> int:
    """Show 'no face', return waitKey code."""
    draw_operator_hud(
        frame,
        mode="IDENTITY",
        command="no face in frame",
        mode_for_keys="identity",
    )
    _draw_flight_status(frame, session_ctx)
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF


def run_identity_preview_loop(
    webcam,
    store: FaceProfileStore,
    *,
    embedding_backend: str,
    match_threshold: float,
    enroll_frames: int = 25,
    window_name: str = "Identity",
    enroll_window_name: str | None = None,
    face_lm: FaceLandmarkerDetector | None = None,
    insight=None,
    config: dict | None = None,
    auto_enroll_if_missing: bool = False,
    session_ctx=None,
) -> str | None:
    eb = str(embedding_backend).lower()
    win = window_name
    enroll_win = enroll_window_name if enroll_window_name else window_name
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    backend_label = "ArcFace (InsightFace)" if eb == "insightface" else "MediaPipe landmarks"
    perf_every, perf_max_w = identity_perf_settings(config or {})
    insight_verifier: CachedInsightFaceVerifier | None = None
    if eb == "insightface" and insight is not None:
        insight_verifier = CachedInsightFaceVerifier(
            insight,
            store,
            match_threshold=float(match_threshold),
            process_every_n_frames=perf_every,
            max_width=perf_max_w,
        )
    print(
        f"[identity] preview — {backend_label} | threshold {match_threshold:.3f}"
        + (
            f" | ArcFace every {perf_every} frames, max width {perf_max_w}px"
            if insight_verifier is not None
            else ""
        )
        + f"\n  N = enroll on screen | L = land | Esc = quit (q only when landed)"
    )
    print(f"[session] switch modes: {MODE_SWITCH_HINT}")

    def _run_enroll_wizard(*, initial_name: str = "") -> bool:
        saved = run_identity_enroll_wizard(
            webcam,
            store,
            embedding_backend=eb,
            enroll_frames=enroll_frames,
            window_name=enroll_win,
            face_lm=face_lm,
            insight=insight,
            initial_name=initial_name,
            reuse_window=(enroll_win == win),
        )
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if saved and insight_verifier is not None:
            insight_verifier.reset()
        return saved is not None

    if auto_enroll_if_missing and not store.exists():
        print("[identity] no profile — on-screen enrollment (Esc to skip)")
        _run_enroll_wizard(initial_name="")

    requested_mode: str | None = None
    try:
        while True:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame = webcam.read_frame()
            if frame is None:
                break

            if not store.exists():
                draw_operator_hud(
                    frame,
                    mode="IDENTITY",
                    command="NO PROFILE",
                    detail="press N to enroll your face",
                    mode_for_keys="identity",
                )
                _draw_flight_status(frame, session_ctx)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                _handle_land_key(key, session_ctx)
                requested_mode = MODE_SWITCH_KEYS.get(key)
                if requested_mode:
                    break
                if is_quit_key(key, _session_drone(session_ctx)):
                    break
                if key in (ord("n"), ord("N")):
                    _run_enroll_wizard(initial_name="")
                continue

            ok = False
            score = 0.0
            stored_name = store.display_name
            x0 = y0 = x1 = y1 = 0

            if eb == "insightface" and insight_verifier is not None:
                result = insight_verifier.process(frame)
                if not result.face_detected:
                    k = _draw_no_face_overlay(frame, win, session_ctx)
                    _handle_land_key(k, session_ctx)
                    requested_mode = MODE_SWITCH_KEYS.get(k)
                    if requested_mode:
                        break
                    if is_quit_key(k, _session_drone(session_ctx)):
                        break
                    if k in (ord("n"), ord("N")):
                        _run_enroll_wizard(initial_name=store.display_name)
                    continue
                ok = result.ok
                score = result.score
                stored_name = result.name
                x0, y0, x1, y1 = result.bbox or (0, 0, 0, 0)
            elif face_lm is not None:
                res = face_lm.process_frame(frame)
                fl = face_lm.first_face_landmarks(res)
                if fl is None:
                    k = _draw_no_face_overlay(frame, win, session_ctx)
                    _handle_land_key(k, session_ctx)
                    requested_mode = MODE_SWITCH_KEYS.get(k)
                    if requested_mode:
                        break
                    if is_quit_key(k, _session_drone(session_ctx)):
                        break
                    if k in (ord("n"), ord("N")):
                        _run_enroll_wizard(initial_name=store.display_name)
                    continue
                emb = face_landmarks_to_embedding(list(fl))
                ok, score, stored_name = store.verify(emb, float(match_threshold))
                x0, y0, x1, y1 = _landmarks_to_bbox(frame, list(fl))
            else:
                draw_operator_hud(
                    frame,
                    mode="IDENTITY",
                    command="ERROR: no face model",
                    mode_for_keys="identity",
                )
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                _handle_land_key(key, session_ctx)
                if is_quit_key(key, _session_drone(session_ctx)):
                    break
                continue

            color = (0, 220, 0) if ok else (0, 80, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            label = f"{stored_name} ({score:.2f})" if ok else f"unknown ({score:.2f})"
            cv2.putText(
                frame,
                label[:40],
                (x0, max(18, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_8,
            )

            draw_operator_hud(
                frame,
                mode="IDENTITY",
                command="MATCH" if ok else "UNKNOWN",
                detail=stored_name if ok else f"score {score:.2f}",
                mode_for_keys="identity",
                command_locked=ok,
            )

            _draw_flight_status(frame, session_ctx)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            _handle_land_key(key, session_ctx)
            requested_mode = MODE_SWITCH_KEYS.get(key)
            if requested_mode:
                break
            if is_quit_key(key, _session_drone(session_ctx)):
                break
            if key in (ord("n"), ord("N")):
                _run_enroll_wizard(initial_name=store.display_name)
    finally:
        if session_ctx is not None:
            session_ctx.finish_mode_window(win, requested_mode)
        else:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
    return requested_mode
