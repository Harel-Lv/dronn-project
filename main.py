import argparse
import cv2
import time

from app.cv2_gui import destroy_all_windows_safe, ensure_opencv_gui_windows
from app.config import load_config
from app.camera.webcam import Webcam
from app.detection.hand_detector import HandDetector
from app.detection.tracking_fsm import TrackingCommandStabilizer
from app.detection.target_tracker import (
    TargetTracker,
    bbox_area_ratio,
    detect_target,
    get_frame_center,
    get_target_center,
)
from app.detection.face_landmarker_detector import FaceLandmarkerDetector
from app.detection.pose_detector import PoseDetector
from app.identity.face_gallery import face_gallery_from_config
from app.identity.face_profile import FaceProfileStore
from app.identity.identity_helpers import resolve_identity_detectors, warn_profile_backend_mismatch
from app.run_gesture_session import (
    run_face_enrollment,
    run_face_enrollment_insightface,
    run_gesture_control_loop,
)
from app.run_identity_preview import run_identity_preview_loop
from app.ui.overlay import draw_tracking_overlay

WINDOW_NAME = "Drone Gesture Control"
TRACKING_WINDOW_NAME = "Target Tracking Preview (SIMULATION ONLY)"

STABLE_FRAMES_REQUIRED = 4  # Require 4 consecutive frames before switching command


def _validate_tracking_config(config: dict) -> None:
    tracking_cfg = config.get("tracking", {})
    required = [
        "center_deadzone",
        "size_min_ratio",
        "size_max_ratio",
        "stable_frames_required",
        "calibration_seconds",
        "calibration_min_samples",
        "calibration_small_factor",
        "calibration_large_factor",
    ]
    missing = [k for k in required if k not in tracking_cfg]
    if missing:
        raise RuntimeError(f"Missing tracking config keys: {', '.join(missing)}")

    # Normalize and validate numeric types from YAML/user edits.
    try:
        tracking_cfg["center_deadzone"] = float(tracking_cfg["center_deadzone"])
        tracking_cfg["size_min_ratio"] = float(tracking_cfg["size_min_ratio"])
        tracking_cfg["size_max_ratio"] = float(tracking_cfg["size_max_ratio"])
        tracking_cfg["stable_frames_required"] = int(tracking_cfg["stable_frames_required"])
        tracking_cfg["calibration_seconds"] = float(tracking_cfg["calibration_seconds"])
        tracking_cfg["calibration_min_samples"] = int(tracking_cfg["calibration_min_samples"])
        tracking_cfg["calibration_small_factor"] = float(tracking_cfg["calibration_small_factor"])
        tracking_cfg["calibration_large_factor"] = float(tracking_cfg["calibration_large_factor"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid tracking config value type: {exc}") from exc

    if tracking_cfg["size_min_ratio"] >= tracking_cfg["size_max_ratio"]:
        raise RuntimeError("Invalid tracking config: size_min_ratio must be smaller than size_max_ratio")

    if tracking_cfg["stable_frames_required"] < 1:
        raise RuntimeError("Invalid tracking config: stable_frames_required must be >= 1")

    if tracking_cfg["calibration_seconds"] <= 0:
        raise RuntimeError("Invalid tracking config: calibration_seconds must be > 0")


def _tracking_preflight(webcam: Webcam, config: dict) -> TargetTracker:
    """
    Runtime health check for tracking mode.
    Fails fast with clear errors before entering tracking loop.
    """
    _validate_tracking_config(config)
    tracking_cfg = config["tracking"]

    tracker = TargetTracker(
        center_deadzone=tracking_cfg["center_deadzone"],
        size_min_ratio=tracking_cfg["size_min_ratio"],
        size_max_ratio=tracking_cfg["size_max_ratio"],
    )

    frame = webcam.read_frame()
    if frame is None:
        raise RuntimeError("Preflight failed: webcam read_frame() returned None")

    # Exercise detector + command pipeline once (safe preview only).
    _bbox, _command = tracker.process_frame(frame)
    return tracker


def run_tracking_loop(webcam: Webcam, config: dict, tracker: TargetTracker | None = None) -> None:
    """Target tracking preview mode - suggests alignment commands, does NOT control drone."""
    cv2.namedWindow(TRACKING_WINDOW_NAME, cv2.WINDOW_NORMAL)
    tracking_cfg = config["tracking"]
    if tracker is None:
        tracker = TargetTracker(
            center_deadzone=tracking_cfg["center_deadzone"],
            size_min_ratio=tracking_cfg["size_min_ratio"],
            size_max_ratio=tracking_cfg["size_max_ratio"],
        )
    stabilizer = TrackingCommandStabilizer(
        stable_frames_required=tracking_cfg["stable_frames_required"]
    )

    # Auto-calibration phase
    print("TRACKING CALIBRATION: look at camera and stay centered...")
    calibration_deadline = time.time() + tracking_cfg["calibration_seconds"]
    samples: list[float] = []

    while time.time() < calibration_deadline:
        try:
            if cv2.getWindowProperty(TRACKING_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return
        except cv2.error:
            return

        frame = webcam.read_frame()
        if frame is None:
            break
        bbox = detect_target(frame, tracker.face_cascade)
        if bbox:
            frame_center = get_frame_center(frame)
            target_center = get_target_center(bbox)
            deadzone_x = int(frame.shape[1] * tracker.center_deadzone)
            if abs(target_center[0] - frame_center[0]) <= deadzone_x:
                samples.append(bbox_area_ratio(frame, bbox))
        cv2.putText(
            frame,
            "Calibrating... keep face in center",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Samples: {len(samples)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(TRACKING_WINDOW_NAME, frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            return

    if len(samples) >= tracking_cfg["calibration_min_samples"]:
        baseline = sum(samples) / len(samples)
        tracker.apply_calibration(
            baseline_ratio=baseline,
            small_factor=tracking_cfg["calibration_small_factor"],
            large_factor=tracking_cfg["calibration_large_factor"],
        )
        print(
            f"CALIBRATED: min_ratio={tracker.size_min_ratio:.3f}, "
            f"max_ratio={tracker.size_max_ratio:.3f}"
        )
    else:
        print("CALIBRATION: not enough samples, using default thresholds")

    while True:
        try:
            if cv2.getWindowProperty(TRACKING_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        frame = webcam.read_frame()
        if frame is None:
            print("FAILED TO READ FRAME")
            break

        bbox, command = tracker.process_frame(frame)
        command = stabilizer.update(command)
        frame_center = get_frame_center(frame)
        target_center = get_target_center(bbox) if bbox else None

        draw_tracking_overlay(frame, bbox, target_center, frame_center, command)
        cv2.imshow(TRACKING_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone Gesture Control / Target Tracking")
    parser.add_argument(
        "--mode",
        choices=["gesture", "fast", "identity", "tracking", "manual", "webcam_faces"],
        default="gesture",
        help=(
            "gesture=hands only | fast=body pose+hands | "
            "identity=face gallery test on PC webcam (fpv_faces + data/faces_gallery) | "
            "tracking=face preview (simulation) | manual=keyboard+mouse PC control (needs --tello --fpv) | "
            "webcam_faces=enrolled face profile + hand gestures (identity gate)"
        ),
    )
    parser.add_argument(
        "--tello",
        action="store_true",
        help="Connect to real DJI Tello (overrides config: drone.enabled=true, backend=tello)",
    )
    parser.add_argument(
        "--fpv",
        action="store_true",
        help="Tello live camera in a second window (pip install av). Works with --tello + gesture/fast/webcam_faces",
    )
    parser.add_argument(
        "--fpv-faces",
        action="store_true",
        help="On --mode manual: match FPV faces to gallery (see config fpv_faces / data/faces_gallery)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default 0); used for gesture/identity/tracking/webcam_faces",
    )
    args = parser.parse_args()
    ensure_opencv_gui_windows()
    config = load_config()
    if args.tello:
        config.setdefault("drone", {})
        config["drone"]["enabled"] = True
        config["drone"]["backend"] = "tello"
    if args.fpv:
        config.setdefault("drone", {})
        config["drone"]["show_tello_fpv"] = True

    if args.fpv_faces:
        config.setdefault("fpv_faces", {})
        config["fpv_faces"]["enabled"] = True

    if args.mode == "manual":
        config.setdefault("drone", {})
        config["drone"]["enabled"] = True
        config["drone"]["backend"] = "tello"
        config["drone"]["show_tello_fpv"] = True

    _dc = config.get("drone", {})
    if args.fpv and not (
        _dc.get("enabled") and str(_dc.get("backend", "")).lower() == "tello"
    ):
        print(
            "NOTE: FPV window needs a real Tello: use --tello or set drone.enabled/backend in config.yaml."
        )

    print("STARTING")

    if args.mode == "webcam_faces":
        # Profile + hand gestures with identity gate (single enrolled person).
        print(
            "MODE: webcam_faces — face profile + gestures; commands only for enrolled person"
        )
        webcam = Webcam(camera_index=args.camera)
        try:
            webcam.open()
            print("WEBCAM OPENED")
            dc = config.get("drone", {})
            if dc.get("enabled") and str(dc.get("backend", "")).lower() == "tello":
                print("DRONE: Tello backend enabled - ensure Wi-Fi connected to drone")
            else:
                print("DRONE: simulation (use --tello or config.yaml for real Tello)")
            detector = HandDetector()
            icfg = config.get("identity", {})
            store = FaceProfileStore(
                str(icfg.get("profile_path", "data/identity/profile.npz")),
                str(icfg.get("display_name", "Owner")),
            )
            enroll_n = int(icfg.get("enroll_frame_count", 25))
            eb, threshold, face_lm, insight_if = resolve_identity_detectors(config)
            warn_profile_backend_mismatch(store, eb)
            if eb == "insightface":
                print("MODE: webcam_faces — ArcFace (InsightFace) + gestures")
            else:
                print("MODE: webcam_faces — needs models/face_landmarker.task (see models/README.md)")
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
                run_gesture_control_loop(
                    webcam,
                    detector,
                    config,
                    mode_label="webcam_faces (known person)",
                    stable_frames_required=STABLE_FRAMES_REQUIRED,
                    face_detector=face_lm if eb == "landmarks" else None,
                    identity_store=store,
                    match_threshold=threshold,
                    insightface_for_identity=insight_if if eb == "insightface" else None,
                    window_name=WINDOW_NAME,
                )
            finally:
                if face_lm is not None:
                    face_lm.close()
                if insight_if is not None:
                    insight_if.close()
                detector.close()
        except FileNotFoundError as exc:
            print(f"WEBCAM_FACES: missing model file or invalid path — {exc}")
        except Exception as exc:
            print(f"WEBCAM_FACES ERROR: {exc}")
        finally:
            webcam.release()
            destroy_all_windows_safe()
        return

    if args.mode == "manual":
        print("MODE: manual — keyboard + mouse PC control (Tello + FPV required)")
        # Lazy import: manual mode needs pynput; other modes should still run without it.
        from app.run_pc_control import run_pc_control_loop

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
            fcfg = config.get("fpv_faces", {})
            try:
                gallery_obj = face_gallery_from_config(config)
            except Exception as exc:
                print(f"FPV FACES: gallery load failed — {exc}")
            eb = str(fcfg.get("embedding_backend", "landmarks")).lower()
            if eb == "insightface":
                try:
                    from app.identity.insightface_backend import InsightFaceBackend

                    face_if = InsightFaceBackend(config)
                    print("FPV FACES: InsightFace (ArcFace) loaded")
                except Exception as exc:
                    print(f"FPV FACES: InsightFace failed — {exc}")
            else:
                try:
                    face_det = FaceLandmarkerDetector()
                except FileNotFoundError as exc:
                    print(f"FPV FACES: face model not found — overlay only. ({exc})")

            run_pc_control_loop(
                config,
                pose_detector=pose,
                face_detector=face_det,
                insightface_backend=face_if,
                face_gallery=gallery_obj,
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
            destroy_all_windows_safe()
        return

    webcam = Webcam(camera_index=args.camera)

    try:
        webcam.open()
        print("WEBCAM OPENED")

        if args.mode == "tracking":
            print("MODE: Target Tracking Preview (SIMULATION - no drone control)")
            try:
                tracker = _tracking_preflight(webcam, config)
                print("TRACKING PREFLIGHT: OK")
            except RuntimeError as exc:
                print(f"TRACKING PREFLIGHT ERROR: {exc}")
                return
            run_tracking_loop(webcam, config, tracker=tracker)

        else:
            dc = config.get("drone", {})
            if dc.get("enabled") and str(dc.get("backend", "")).lower() == "tello":
                print("DRONE: Tello backend enabled - ensure Wi-Fi connected to drone")
            else:
                print("DRONE: simulation (use --tello or config.yaml for real Tello)")

            if args.mode == "identity":
                print(
                    "MODE: identity — single-person recognition (name shown only on match)"
                )
                icfg = config.get("identity", {})
                store = FaceProfileStore(
                    str(icfg.get("profile_path", "data/identity/profile.npz")),
                    str(icfg.get("display_name", "Owner")),
                )
                enroll_n = int(icfg.get("enroll_frame_count", 25))
                identity_win = "Identity"
                eb, threshold, face_lm, insight_if = resolve_identity_detectors(config)
                warn_profile_backend_mismatch(store, eb)
                try:
                    if not store.exists():
                        print(
                            "MODE: identity — no profile found; short enrollment before preview "
                            f"({enroll_n} frames)"
                        )
                        if eb == "insightface":
                            run_face_enrollment_insightface(
                                webcam,
                                insight_if,
                                store,
                                enroll_frames=enroll_n,
                                display_name=store.display_name,
                                window_name="Identity — enroll",
                            )
                        else:
                            run_face_enrollment(
                                webcam,
                                face_lm,
                                store,
                                enroll_frames=enroll_n,
                                display_name=store.display_name,
                                window_name="Identity — enroll",
                            )
                    run_identity_preview_loop(
                        webcam,
                        store,
                        embedding_backend=eb,
                        match_threshold=threshold,
                        enroll_frames=enroll_n,
                        window_name=identity_win,
                        enroll_window_name="Identity — enroll",
                        face_lm=face_lm,
                        insight=insight_if,
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
            else:
                detector = HandDetector()

                if args.mode == "gesture":
                    print("MODE: gesture — hands only (no pose / no identity gate)")
                    try:
                        run_gesture_control_loop(
                            webcam,
                            detector,
                            config,
                            mode_label="gesture (hands)",
                            stable_frames_required=STABLE_FRAMES_REQUIRED,
                            window_name=WINDOW_NAME,
                        )
                    finally:
                        detector.close()

                elif args.mode == "fast":
                    print("MODE: fast — body pose + hands (needs models/pose_landmarker_lite.task)")
                    pose = PoseDetector()
                    try:
                        run_gesture_control_loop(
                            webcam,
                            detector,
                            config,
                            mode_label="fast (pose+hands)",
                            stable_frames_required=STABLE_FRAMES_REQUIRED,
                            pose_detector=pose,
                            window_name=WINDOW_NAME,
                        )
                    finally:
                        pose.close()
                        detector.close()
    except KeyboardInterrupt:
        pass
    finally:
        webcam.release()
        destroy_all_windows_safe()


if __name__ == "__main__":
    main()
