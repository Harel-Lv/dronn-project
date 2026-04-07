from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import warnings

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "tracking": {
        "center_deadzone": 0.15,
        "size_min_ratio": 0.03,
        "size_max_ratio": 0.25,
        "stable_frames_required": 3,
        "calibration_seconds": 8,
        "calibration_min_samples": 20,
        "calibration_small_factor": 0.75,
        "calibration_large_factor": 1.35,
    },
    "drone": {
        "enabled": False,
        "backend": "simulated",
        "tello_move_distance_cm": 30,
        "tello_host": None,
        "show_tello_fpv": False,
        "follow_pose_every_n_frames": 2,
        "follow_pose_max_width": 480,
        # חיבור Tello (לפני טיסה — עמידות לרשת)
        "tello_connect_retries": 3,
        "tello_connect_retry_delay_sec": 1.5,
        # FPV: אם אין פריים מהרחפן יותר מ-X שניות — מאפסים RC (0 = כבוי)
        "fpv_stale_seconds": 5.0,
    },
    "identity": {
        "profile_path": "data/identity/profile.npz",
        "display_name": "Owner",
        "match_threshold": 0.88,
        "enroll_frame_count": 25,
    },
    "fpv_faces": {
        "enabled": False,
        "gallery_dir": "data/faces_gallery",
        "embedding_backend": "landmarks",
        "match_threshold": 0.93,
        "match_threshold_insightface": 0.42,
        "insightface_det_size": 640,
        "insightface_ctx_id": -1,
        "insightface_process_every_n_frames": 2,
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
        # כשל מצלמת מחשב במצב מחוות: ספירת פריימים ריקים ברצף לפני עצירה
        "max_consecutive_bad_frames": 30,
        # אם True + רחפן אמיתי — נחיתה אחרי מכסת bad frames (זהירות: false positives)
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
    merged_identity = DEFAULT_CONFIG["identity"].copy()
    merged_identity.update(data.get("identity", {}))
    merged["identity"] = merged_identity
    merged_fpv = DEFAULT_CONFIG["fpv_faces"].copy()
    merged_fpv.update(data.get("fpv_faces", {}))
    merged["fpv_faces"] = merged_fpv
    merged_gestures = DEFAULT_CONFIG["gestures"].copy()
    merged_gestures.update(data.get("gestures", {}))
    merged["gestures"] = merged_gestures
    return merged
