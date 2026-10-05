"""Session dispatch — shared by CLI (main.py) and launcher UI."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

from app.cv2_gui import destroy_all_windows_safe, ensure_opencv_gui_windows
from app.config import load_config
from app.detection.face_landmarker_detector import FaceLandmarkerDetector
from app.detection.hand_detector import HandDetector
from app.detection.pose_detector import PoseDetector
from app.identity.face_gallery import face_gallery_from_config
from app.identity.face_profile import FaceProfileStore
from app.identity.identity_helpers import resolve_identity_detectors, warn_profile_backend_mismatch
from app.run_body_follow import run_body_follow_loop
from app.run_chase_mode import run_chase_loop
from app.run_gesture_session import (
    run_face_enrollment,
    run_face_enrollment_insightface,
    run_gesture_control_loop,
)
from app.run_identity_preview import run_identity_preview_loop
from app.session_context import SessionContext
from app.session_modes import MODE_CHOICES, MODE_SWITCH_HINT, SESSION_MAIN_WINDOW

WINDOW_NAME = SESSION_MAIN_WINDOW
STABLE_FRAMES_REQUIRED = 4


@dataclass
class SessionOptions:
    mode: str = "manual"
    tello: bool = False
    fpv: bool = False
    fpv_faces: bool = False
    camera: int = 0

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> SessionOptions:
        return cls(
            mode=str(args.mode),
            tello=bool(args.tello),
            fpv=bool(args.fpv),
            fpv_faces=bool(args.fpv_faces),
            camera=int(args.camera),
        )


def apply_options_to_config(config: dict, opts: SessionOptions) -> dict:
    if opts.tello:
        config.setdefault("drone", {})
        config["drone"]["enabled"] = True
        config["drone"]["backend"] = "tello"
    if opts.fpv:
        config.setdefault("drone", {})
        config["drone"]["show_tello_fpv"] = True
    if opts.fpv_faces:
        config.setdefault("fpv_faces", {})
        config["fpv_faces"]["enabled"] = True
    if opts.mode == "manual" and opts.tello:
        config.setdefault("drone", {})
        config["drone"]["enabled"] = True
        config["drone"]["backend"] = "tello"
        config["drone"]["show_tello_fpv"] = True
    if opts.mode == "tracking":
        config.setdefault("drone", {})
        config["drone"]["show_tello_fpv"] = True
        if opts.tello:
            config["drone"]["enabled"] = True
            config["drone"]["backend"] = "tello"
    if opts.mode == "chase":
        config.setdefault("drone", {})
        config["drone"]["show_tello_fpv"] = True
        if opts.tello:
            config["drone"]["enabled"] = True
            config["drone"]["backend"] = "tello"
    return config


def _next_mode_from_result(current_mode: str, result: str | None) -> tuple[bool, str]:
    if result is None:
        return False, current_mode
    nm = str(result).strip().lower()
    if nm in MODE_CHOICES and nm != current_mode:
        return True, nm
    return False, current_mode


def _run_single_mode(
    current_mode: str,
    cfg: dict,
    loop_opts: SessionOptions,
    ctx: SessionContext,
) -> str | None:
    """Run one mode loop; shared drone/webcam stay open for soft switches."""
    next_mode: str | None = None

    if current_mode == "webcam_faces":
        print(
            "MODE: webcam_faces — face profile + gestures; commands only for enrolled person"
        )
        webcam = ctx.ensure_webcam(loop_opts.camera)
        detector = HandDetector()
        icfg = cfg.get("identity", {})
        store = FaceProfileStore(
            str(icfg.get("profile_path", "data/identity/profile.npz")),
            str(icfg.get("display_name", "Owner")),
        )
        enroll_n = int(icfg.get("enroll_frame_count", 25))
        eb, threshold, face_lm, insight_if = resolve_identity_detectors(cfg)
        warn_profile_backend_mismatch(store, eb)
        if eb == "insightface":
            print("MODE: webcam_faces — ArcFace (InsightFace) + gestures")
        else:
            print(
                "MODE: webcam_faces — needs models/face_landmarker.task (see models/README.md)"
            )
        try:
            if not store.exists():
                if eb == "insightface":
                    run_face_enrollment_insightface(
                        webcam,
                        insight_if,
                        store,
                        enroll_frames=enroll_n,
                        display_name=store.display_name,
                        window_name=WINDOW_NAME,
                    )
                else:
                    run_face_enrollment(
                        webcam,
                        face_lm,
                        store,
                        enroll_frames=enroll_n,
                        display_name=store.display_name,
                        window_name=WINDOW_NAME,
                    )
            next_mode = run_gesture_control_loop(
                webcam,
                detector,
                cfg,
                mode_label="webcam_faces (known person)",
                stable_frames_required=STABLE_FRAMES_REQUIRED,
                face_detector=face_lm if eb == "landmarks" else None,
                identity_store=store,
                match_threshold=threshold,
                insightface_for_identity=insight_if if eb == "insightface" else None,
                window_name=WINDOW_NAME,
                session_ctx=ctx,
            )
        except FileNotFoundError as exc:
            print(f"WEBCAM_FACES: missing model file or invalid path — {exc}")
        except Exception as exc:
            print(f"WEBCAM_FACES ERROR: {exc}")
        finally:
            if face_lm is not None:
                face_lm.close()
            if insight_if is not None:
                insight_if.close()
            detector.close()
        return next_mode

        if current_mode == "manual":
            print("MODE: manual — keyboard + mouse PC control (Tello + FPV required)")
            from app.run_pc_control import run_pc_control_loop

            ctx.ensure_webcam(loop_opts.camera)
        pose = None
        face_det = None
        face_if = None
        try:
            try:
                pose = PoseDetector()
                print("FOLLOW mode: pose model loaded (Tab long-press to switch)")
            except FileNotFoundError:
                print("FOLLOW mode: pose model missing — Tab long-press disabled")

            gallery_obj = None
            fcfg = cfg.get("fpv_faces", {})
            if bool(fcfg.get("enabled", False)):
                try:
                    gallery_obj = face_gallery_from_config(cfg)
                except Exception as exc:
                    print(f"FPV FACES: gallery load failed — {exc}")
            eb = str(fcfg.get("embedding_backend", "landmarks")).lower()
            if eb == "insightface":
                try:
                    from app.identity.insightface_backend import InsightFaceBackend

                    face_if = InsightFaceBackend(cfg)
                    print("FPV FACES: InsightFace (ArcFace) loaded")
                except Exception as exc:
                    print(f"FPV FACES: InsightFace failed — {exc}")
            else:
                try:
                    face_det = FaceLandmarkerDetector()
                except FileNotFoundError as exc:
                    print(f"FPV FACES: face model not found — overlay only. ({exc})")

            next_mode = run_pc_control_loop(
                cfg,
                pose_detector=pose,
                face_detector=face_det,
                insightface_backend=face_if,
                face_gallery=gallery_obj,
                session_ctx=ctx,
            )
        except Exception as exc:
            print(f"MANUAL MODE ERROR: {exc}")
        finally:
            if pose is not None:
                pose.close()
            if face_det is not None:
                face_det.close()
            if face_if is not None:
                face_if.close()
        return next_mode

    if current_mode == "chase":
        print("MODE: chase — any face = red target; hold SPACE to rush (no enrollment)")
        webcam = ctx.ensure_webcam(loop_opts.camera)
        insight_if = None
        try:
            from app.identity.insightface_backend import InsightFaceBackend

            insight_if = InsightFaceBackend(cfg)
            next_mode = run_chase_loop(cfg, webcam=webcam, insight=insight_if, session_ctx=ctx)
        except RuntimeError as exc:
            print(f"CHASE: InsightFace not ready — {exc}")
        except Exception as exc:
            print(f"CHASE ERROR: {exc}")
        finally:
            if insight_if is not None:
                insight_if.close()
        return next_mode

    if current_mode == "tracking":
        print("MODE: tracking — body follow on FPV (pose lock at entry)")
        webcam = ctx.ensure_webcam(loop_opts.camera)
        pose: PoseDetector | None = None
        try:
            pose = PoseDetector()
            next_mode = run_body_follow_loop(
                cfg, webcam=webcam, pose_detector=pose, session_ctx=ctx
            )
        except FileNotFoundError as exc:
            print(f"TRACKING: missing pose model — {exc}")
        except Exception as exc:
            print(f"TRACKING ERROR: {exc}")
        finally:
            if pose is not None:
                pose.close()
        return next_mode

    webcam = ctx.ensure_webcam(loop_opts.camera)

    if current_mode == "identity":
        print("MODE: identity — single-person recognition (name shown only on match)")
        icfg = cfg.get("identity", {})
        store = FaceProfileStore(
            str(icfg.get("profile_path", "data/identity/profile.npz")),
            str(icfg.get("display_name", "Owner")),
        )
        enroll_n = int(icfg.get("enroll_frame_count", 25))
        identity_win = SESSION_MAIN_WINDOW
        eb, threshold, face_lm, insight_if = resolve_identity_detectors(cfg)
        warn_profile_backend_mismatch(store, eb)
        try:
            next_mode = run_identity_preview_loop(
                webcam,
                store,
                embedding_backend=eb,
                match_threshold=threshold,
                enroll_frames=enroll_n,
                window_name=identity_win,
                enroll_window_name=identity_win,
                face_lm=face_lm,
                insight=insight_if,
                config=cfg,
                auto_enroll_if_missing=True,
                session_ctx=ctx,
            )
        except FileNotFoundError as exc:
            print(f"IDENTITY: missing model file or invalid path — {exc}")
        except Exception as exc:
            print(f"IDENTITY ERROR: {exc}")
        finally:
            if face_lm is not None:
                face_lm.close()
            if insight_if is not None:
                insight_if.close()
        return next_mode

    if current_mode == "gesture":
        print("MODE: gesture — hands only (no pose / no identity gate)")
        detector = HandDetector()
        try:
            next_mode = run_gesture_control_loop(
                webcam,
                detector,
                cfg,
                mode_label="gesture (hands)",
                stable_frames_required=STABLE_FRAMES_REQUIRED,
                window_name=WINDOW_NAME,
                session_ctx=ctx,
            )
        finally:
            detector.close()
        return next_mode

    print(f"Unknown mode: {current_mode!r}")
    return None


def run_session(opts: SessionOptions, config: dict | None = None) -> None:
    """Run one application session (gesture, identity, manual, etc.)."""
    ensure_opencv_gui_windows()
    base_config = config or load_config()
    ctx = SessionContext(opts=opts, base_config=base_config)
    current_mode = opts.mode
    try:
        while True:
            loop_opts = SessionOptions(
                mode=current_mode,
                tello=opts.tello,
                fpv=opts.fpv,
                fpv_faces=opts.fpv_faces,
                camera=opts.camera,
            )
            cfg = apply_options_to_config(copy.deepcopy(base_config), loop_opts)

            _dc = cfg.get("drone", {})
            if loop_opts.fpv and not (
                _dc.get("enabled") and str(_dc.get("backend", "")).lower() == "tello"
            ):
                print(
                    "NOTE: FPV window needs a real Tello: use --tello or set drone.enabled/backend in config.yaml."
                )

            print(f"STARTING mode={current_mode}")
            print(f"[session] switch modes: {MODE_SWITCH_HINT}")

            try:
                next_mode = _run_single_mode(current_mode, cfg, loop_opts, ctx)
            except KeyboardInterrupt:
                break

            switched, current_mode = _next_mode_from_result(current_mode, next_mode)
            if not switched:
                break
            ctx.prepare_mode_switch()
            print(f"[session] soft switch → {current_mode} (drone stays connected)")
            ctx.show_mode_switch_overlay(current_mode)
            ctx.preserve_main_window = False
    finally:
        ctx.shutdown()
