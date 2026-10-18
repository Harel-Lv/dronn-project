from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import warnings

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "tracking": {
        "center_deadzone": 0.12,
        "size_min_ratio": 0.04,
        "size_max_ratio": 0.22,
        "calibration_seconds": 6.0,
        "calibration_min_samples": 12,
        "calibration_small_factor": 0.78,
        "calibration_large_factor": 1.32,
        "process_every_n_frames": 2,
        "max_width": 480,
        "max_rc_speed": 48,
        "require_identity_profile": False,
        # Keep the person vertically centered by climbing/descending inside a height band
        "vertical_enabled": True,
        "vertical_gain": 40.0,
        "vertical_deadzone": 0.15,
        "min_height_cm": 60,
        "max_height_cm": 220,
        # Damping on how fast the target moves in frame (higher = less overshoot)
        "derivative_gain": 10.0,
        # RC smoothing: 0..1 (1 = no smoothing) and max change per frame per axis
        "smoothing_alpha": 0.4,
        "max_rc_step": 15,
    },
    "chase": {
        "process_every_n_frames": 2,
        "max_width": 480,
        "max_rc_speed": 85,
        "hold_seconds": 0.35,
        "stop_size_ratio": 0.32,
        "derivative_gain": 10.0,
        "smoothing_alpha": 0.4,
        "max_rc_step": 20,
    },
    "rt_control": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 9999,
        "service_bind": "0.0.0.0",
        "control_hz": 20.0,
        "loop_hz": 20.0,
        "watchdog_hover_ms": 500,
        "watchdog_safe_ms": 2000,
        "stale_packet_ms": 2000,
    },
    "drone": {
        "enabled": False,
        "backend": "simulated",
        "tello_move_distance_cm": 30,
        "tello_host": None,
        "show_tello_fpv": False,
        "follow_pose_every_n_frames": 2,
        "follow_pose_max_width": 480,
        # Tello connection (preflight — network resilience)
        "tello_connect_retries": 3,
        "tello_connect_retry_delay_sec": 1.5,
        "tello_state_wait_sec": 8.0,
        "tello_response_timeout_sec": 8.0,
        # FPV: no drone frame for more than X seconds -> zero RC (0 = off)
        "fpv_stale_seconds": 5.0,
        # FPV frame unchanged for this long = frozen stream (treated as no frame)
        "fpv_freeze_seconds": 1.0,
        # While airborne: hover RC if no command for this long (Tello auto-lands after 15 s,
        # and keeps executing the last RC until a new one arrives)
        "keepalive_idle_sec": 1.5,
        # In flight: warn at this battery %, auto-land at that %
        "battery_inflight_warn_percent": 20,
        "battery_autoland_percent": 10,
        # No telemetry (UDP 8890) for this long while connected = LINK LOST on the HUD
        "link_lost_seconds": 2.0,
        # Highest motor-board temperature that counts as "too hot" (Tello refuses takeoff)
        "temp_warn_c": 85,
        # Auto-land when the session ends while the drone is still airborne
        "land_on_disconnect": True,
    },
    "identity": {
        "profile_path": "data/identity/profile.npz",
        "display_name": "Owner",
        "embedding_backend": "insightface",
        "match_threshold": 0.88,
        "match_threshold_insightface": 0.42,
        "enroll_frame_count": 25,
        "insightface_det_size": 320,
        "insightface_process_every_n_frames": 3,
        "insightface_max_width": 480,
    },
    "fpv_faces": {
        "enabled": False,
        "gallery_dir": "data/faces_gallery",
        "embedding_backend": "landmarks",
        "match_threshold": 0.93,
        "match_threshold_insightface": 0.42,
        "insightface_det_size": 320,
        "insightface_ctx_id": -1,
        "insightface_process_every_n_frames": 2,
        "insightface_max_width": 480,
        "min_best_vs_second_gap": 0.045,
        "duplicate_registration_threshold": None,
        "allowed_names": None,
        "include_identity_profile": True,
    },
    "gestures": {
        "dominant_hand": "any",
        "hand_calibration_seconds": 2.5,
        "hand_calibration_min_open_frames": 8,
        "stable_frames_normal": 4,
        "stable_frames_critical": 8,
        "finger_extension_base_ratio": 1.12,
        "thumb_extension_base_ratio": 1.08,
        "palm_scale_clamp_low": 0.72,
        "palm_scale_clamp_high": 1.48,
        # PC webcam failure in gesture mode: consecutive empty frames before stopping
        "max_consecutive_bad_frames": 30,
        # If True with a real drone — land after max bad frames (careful: false positives)
        "land_on_camera_lost": False,
    },
}


def load_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config.yaml"
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        warnings.warn(f"Failed to parse config file at {path}. Using defaults.")
        return copy.deepcopy(DEFAULT_CONFIG)

    if not isinstance(data, dict):
        warnings.warn(f"Invalid config format in {path}. Using defaults.")
        return copy.deepcopy(DEFAULT_CONFIG)

    merged = DEFAULT_CONFIG.copy()
    merged_tracking = DEFAULT_CONFIG["tracking"].copy()
    merged_tracking.update(data.get("tracking", {}))
    merged["tracking"] = merged_tracking
    merged_drone = DEFAULT_CONFIG["drone"].copy()
    merged_drone.update(data.get("drone", {}))
    merged["drone"] = merged_drone
    merged_rt = DEFAULT_CONFIG["rt_control"].copy()
    merged_rt.update(data.get("rt_control", {}))
    merged["rt_control"] = merged_rt
    merged_identity = DEFAULT_CONFIG["identity"].copy()
    merged_identity.update(data.get("identity", {}))
    merged["identity"] = merged_identity
    merged_fpv = DEFAULT_CONFIG["fpv_faces"].copy()
    merged_fpv.update(data.get("fpv_faces", {}))
    merged["fpv_faces"] = merged_fpv
    merged_gestures = DEFAULT_CONFIG["gestures"].copy()
    merged_gestures.update(data.get("gestures", {}))
    merged["gestures"] = merged_gestures
    merged_chase = DEFAULT_CONFIG["chase"].copy()
    merged_chase.update(data.get("chase", {}))
    merged["chase"] = merged_chase
    return merged
