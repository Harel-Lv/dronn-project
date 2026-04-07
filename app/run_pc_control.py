"""Manual keyboard control + optional hand gestures + FPV follow."""
from __future__ import annotations

import time

import cv2
import numpy as np

from app.camera.webcam import Webcam
from app.detection.hand_detector import HandDetector
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
from app.gestures.motion_analyzer import MotionAnalyzer
from app.input.pc_controller import PcInputController
from app.identity.face_gallery import save_embedding_to_gallery_file
from app.identity.face_profile import face_landmarks_to_embedding
from app.ui.hud_session import draw_manual_mode_hud

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


def _blank_frame(text: str) -> np.ndarray:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return img


def _follow_lr_yaw(
    pose_detector,
    frame_bgr: np.ndarray,
    max_width: int,
) -> tuple[int, int]:
    h, w = frame_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return 0, 0
    scale = min(1.0, float(max_width) / float(w))
    small = cv2.resize(
        frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    res = pose_detector.process_frame(small)
    lm = pose_detector.landmarks_to_pixel_dict(small, res)
    if not lm:
        return 0, 0
    if 23 in lm and 24 in lm:
        cx = (lm[23][0] + lm[24][0]) // 2
    elif 0 in lm:
        cx = lm[0][0]
    else:
        return 0, 0
    fw = small.shape[1]
    err = (cx - fw * 0.5) / max(fw * 0.5, 1.0)
    if abs(err) < 0.06:
        return 0, 0
    lr = int(np.clip(-err * 72.0, -100, 100))
    yaw = int(np.clip(err * 58.0, -100, 100))
    return lr, yaw


def run_pc_control_loop(
    config: dict,
    *,
    pose_detector=None,
    face_detector=None,
    insightface_backend=None,
    face_gallery=None,
) -> None:
    win = "Tello FPV — manual (Esc | T/L | Tab=FOLLOW | H=hand-gestures)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    drone = DroneController(config)
    drone.connect()
    inp = PcInputController()
    try:
        inp.start()
    except Exception:
        try:
            drone.disconnect()
        except Exception:
            pass
        raise

    dcfg = config.get("drone", {})
    follow_every = max(1, int(dcfg.get("follow_pose_every_n_frames", 2)))
    follow_max_w = max(160, int(dcfg.get("follow_pose_max_width", 480)))

    fcfg = config.get("fpv_faces", {})
    face_every = max(1, int(fcfg.get("insightface_process_every_n_frames", 2)))
    faces_available = face_gallery is not None and (
        insightface_backend is not None or face_detector is not None
    )

    gcfg = config.get("gestures", {}) or {}
    dominant_hand = str(gcfg.get("dominant_hand", "any"))
    hand_cam_idx = int(dcfg.get("manual_gesture_camera_index", 0))

    frame_idx = 0
    last_face_label = ""
    face_id_mode = False
    last_embedding: np.ndarray | None = None
    last_face_name: str | None = None
    last_face_score = 0.0
    battery_pct: int | None = None
    fpv_on = bool(dcfg.get("show_tello_fpv", False))
    stale_sec = float(dcfg.get("fpv_stale_seconds", 5.0))
    last_good_fpv = time.monotonic()
    fpv_stale_warned = False

    hand_mode = False
    hand_seen = False
    hand_cam: Webcam | None = None
    hand_det: HandDetector | None = None
    hand_analyzer: MotionAnalyzer | None = None
    hand_stab = GestureIntentStabilizer(
        stable_frames_normal=int(gcfg.get("stable_frames_normal", 4)),
        stable_frames_critical=int(gcfg.get("stable_frames_critical", 8)),
        initial_intent=HOVER,
    )
    prev_stable = HOVER

    try:
        while True:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if inp.quit_requested:
                break

            raw_fpv = drone.read_fpv_frame()
            if raw_fpv is not None:
                frame = raw_fpv
                last_good_fpv = time.monotonic()
                fpv_stale_warned = False
            else:
                frame = _blank_frame("No FPV frame — check stream / Wi-Fi")

            if drone.is_live_tello:
                battery_pct = drone.get_battery_percent(min_interval_sec=1.0)

            lr, fb, ud, yaw = inp.compute_manual_rc()

            if inp.face_id_toggle_requested:
                inp.face_id_toggle_requested = False
                if faces_available:
                    face_id_mode = not face_id_mode
                    print(f"[manual] face-id mode = {face_id_mode} (toggle)")
                else:
                    print("[manual] face-id not available (gallery/backend missing)")

            if inp.face_save_requested:
                inp.face_save_requested = False
                if not face_id_mode:
                    print("[manual] N ignored: face-id is OFF.")
                elif last_embedding is None:
                    print("[manual] N ignored: no face embedding yet.")
                elif last_face_name is not None:
                    print(f"[manual] N ignored: already recognized as {last_face_name}.")
                else:
                    try:
                        name = input("שם לשמירה במאגר: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        name = ""
                    if not name:
                        print("[manual] save cancelled.")
                    else:
                        out = save_embedding_to_gallery_file(
                            str(face_gallery.gallery_dir),
                            last_embedding,
                            name,
                            embedding_backend=str(fcfg.get("embedding_backend", "landmarks")),
                        )
                        if out is None:
                            print("[manual] save failed.")
                        else:
                            face_gallery.add_from_npz_file(out, display_name=name)
                            last_face_label = f"{name} ({last_face_score:.2f})"
                            print(f"[manual] saved: {out.name}")

            if inp.hand_mode_toggle_requested:
                inp.hand_mode_toggle_requested = False
                hand_mode = not hand_mode
                if hand_mode:
                    try:
                        if hand_det is not None:
                            hand_det.close()
                            hand_det = None
                        if hand_cam is not None:
                            hand_cam.release()
                            hand_cam = None

                        from app.run_gesture_session import _run_open_palm_calibration

                        candidate_idxs = [hand_cam_idx]
                        for ci in (0, 1, 2):
                            if ci not in candidate_idxs:
                                candidate_idxs.append(ci)
                        palm_ref = None
                        calib_meta = {"status": "failed", "samples_ok": 0, "min_samples": 0}
                        selected_idx = None
                        for ci in candidate_idxs:
                            try:
                                hand_cam = Webcam(camera_index=ci)
                                hand_cam.open()
                                hand_det = HandDetector()
                                print(f"[manual] hand camera probe index={ci}")
                                palm_ref, calib_meta = _run_open_palm_calibration(
                                    hand_cam,
                                    hand_det,
                                    window_name="Manual hand calibration",
                                    gcfg=gcfg,
                                    dominant_hand=dominant_hand,
                                    close_window_on_exit=True,
                                )
                                selected_idx = ci
                                # Once a camera opens, continue with it even if calibration
                                # has low samples; in-session detection can still succeed.
                                break
                            except Exception as probe_exc:
                                print(f"[manual] hand camera index={ci} failed: {probe_exc}")
                                if hand_det is not None:
                                    try:
                                        hand_det.close()
                                    except Exception:
                                        pass
                                    hand_det = None
                                if hand_cam is not None:
                                    hand_cam.release()
                                    hand_cam = None
                                continue
                        if selected_idx is None:
                            raise RuntimeError("no usable hand camera found")
                        hand_analyzer = MotionAnalyzer(
                            use_pose=False,
                            reference_palm_span=palm_ref,
                            finger_base_ratio=float(gcfg.get("finger_extension_base_ratio", 1.12)),
                            thumb_base_ratio=float(gcfg.get("thumb_extension_base_ratio", 1.08)),
                            palm_scale_clamp=(
                                float(gcfg.get("palm_scale_clamp_low", 0.72)),
                                float(gcfg.get("palm_scale_clamp_high", 1.48)),
                            ),
                        )
                        hand_stab = GestureIntentStabilizer(
                            stable_frames_normal=int(gcfg.get("stable_frames_normal", 4)),
                            stable_frames_critical=int(gcfg.get("stable_frames_critical", 8)),
                            initial_intent=HOVER,
                        )
                        prev_stable = HOVER
                        print(
                            "[manual] hand-gesture mode = True "
                            f"(cam={selected_idx}, status={calib_meta.get('status')}, "
                            f"samples={calib_meta.get('samples_ok')}/{calib_meta.get('min_samples')})"
                        )
                    except Exception as exc:
                        print(f"[manual] hand-gesture mode failed: {exc}")
                        hand_mode = False
                        if hand_det is not None:
                            try:
                                hand_det.close()
                            except Exception:
                                pass
                            hand_det = None
                        if hand_cam is not None:
                            hand_cam.release()
                            hand_cam = None
                        hand_analyzer = None
                else:
                    print("[manual] hand-gesture mode = False (back to keyboard)")
                    if hand_det is not None:
                        try:
                            hand_det.close()
                        except Exception:
                            pass
                        hand_det = None
                    if hand_cam is not None:
                        hand_cam.release()
                        hand_cam = None
                    hand_analyzer = None

            if (not hand_mode) and inp.follow_mode and pose_detector is not None:
                if frame_idx % follow_every == 0:
                    flr, fyaw = _follow_lr_yaw(
                        pose_detector, frame, follow_max_w
                    )
                    lr, yaw = flr, fyaw

            if hand_mode and hand_det is not None and hand_analyzer is not None and hand_cam is not None:
                hframe = hand_cam.read_frame()
                if hframe is not None:
                    hres = hand_det.process_frame(hframe)
                    pairs = hand_det.extract_hands_with_handedness(hframe, hres)
                    chosen = select_dominant_hand_landmarks(pairs, dominant_hand)
                    hand_seen = chosen is not None
                    hands = [chosen] if chosen is not None else []
                    g = hand_analyzer.analyze(hands, None)
                    raw_intent = gesture_to_drone_intent(g)
                    if raw_intent == NO_GESTURE:
                        raw_intent = HOVER
                    stable = hand_stab.update(raw_intent)
                    if stable != prev_stable:
                        print(
                            f"[manual/hand] intent: gesture={g} raw={raw_intent} "
                            f"stable={stable} (hands={'yes' if chosen is not None else 'no'})"
                        )

                    if stable == TAKEOFF and prev_stable != TAKEOFF:
                        try:
                            drone.send_rc(0, 0, 0, 0)
                            drone.takeoff()
                            print("[manual/hand] takeoff")
                        except Exception as exc:
                            print(f"[manual/hand] takeoff: {exc}")
                        lr, fb, ud, yaw = 0, 0, 0, 0
                    elif stable == LAND and prev_stable != LAND:
                        try:
                            drone.send_rc(0, 0, 0, 0)
                            drone.land()
                            print("[manual/hand] land")
                        except Exception as exc:
                            print(f"[manual/hand] land: {exc}")
                        lr, fb, ud, yaw = 0, 0, 0, 0
                    elif stable in _INTENT_RC:
                        lr, fb, ud, yaw = _INTENT_RC[stable]
                    else:
                        lr, fb, ud, yaw = 0, 0, 0, 0
                    prev_stable = stable
                else:
                    hand_seen = False
                    lr, fb, ud, yaw = 0, 0, 0, 0
            else:
                hand_seen = False

            fpv_stale = (
                fpv_on
                and stale_sec > 0
                and drone.is_live_tello
                and raw_fpv is None
                and (time.monotonic() - last_good_fpv) >= stale_sec
            )
            if fpv_stale:
                lr, fb, ud, yaw = 0, 0, 0, 0
                if not fpv_stale_warned:
                    print(
                        "[manual] FPV לא מתעדכן — מאפסים RC "
                        f"(>{stale_sec:.0f}s; fpv_stale_seconds ב-config)"
                    )
                    fpv_stale_warned = True

            if face_id_mode and faces_available and frame_idx % face_every == 0:
                if insightface_backend is not None:
                    emb, _bbox = insightface_backend.embed(frame)
                    insightface_backend.draw_bbox(frame, _bbox)
                    last_embedding = emb
                    if emb is not None:
                        name, sc = face_gallery.best_match(emb)
                        last_face_name = name
                        last_face_score = float(sc)
                        last_face_label = f"{name or 'לא ידוע'} ({sc:.2f})"
                    else:
                        last_face_name = None
                        last_face_score = 0.0
                        last_face_label = "לא זוהתה פנים"
                elif face_detector is not None:
                    res = face_detector.process_frame(frame)
                    fl = face_detector.first_face_landmarks(res)
                    emb = (
                        face_landmarks_to_embedding(list(fl))
                        if fl is not None
                        else None
                    )
                    last_embedding = emb
                    if emb is not None:
                        name, sc = face_gallery.best_match(emb)
                        last_face_name = name
                        last_face_score = float(sc)
                        last_face_label = f"{name or 'לא ידוע'} ({sc:.2f})"
                    else:
                        last_face_name = None
                        last_face_score = 0.0
                        last_face_label = "לא זוהתה פנים"
            elif not face_id_mode:
                last_face_label = "face-id: off (I to enable)"

            if inp.takeoff_requested:
                inp.takeoff_requested = False
                try:
                    drone.send_rc(0, 0, 0, 0)
                    drone.takeoff()
                    print("[manual] takeoff")
                except Exception as exc:
                    print(f"[manual] takeoff: {exc}")

            if inp.emergency_land:
                inp.emergency_land = False
                try:
                    drone.send_rc(0, 0, 0, 0)
                    drone.land()
                except Exception as exc:
                    print(f"[manual] emergency land: {exc}")

            drone.send_rc(lr, fb, ud, yaw)

            draw_manual_mode_hud(
                frame,
                is_live_tello=drone.is_live_tello,
                follow_mode=inp.follow_mode,
                control_mode="HANDS" if hand_mode else "KEYBOARD",
                rc=(lr, fb, ud, yaw),
                last_face=(
                    f"{last_face_label or '—'} | hand={'yes' if hand_seen else 'no'}"
                    if hand_mode
                    else (last_face_label or "—")
                ),
                fpv_stale=fpv_stale,
                battery_pct=battery_pct,
            )
            cv2.imshow(win, frame)
            cv2.waitKey(1)
            frame_idx += 1
            time.sleep(0.01)
    finally:
        if hand_det is not None:
            try:
                hand_det.close()
            except Exception:
                pass
        if hand_cam is not None:
            hand_cam.release()
        inp.stop()
        try:
            drone.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        drone.disconnect()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
