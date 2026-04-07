"""Body pose via MediaPipe Pose Landmarker (lite)."""
from __future__ import annotations

import os

import cv2
import numpy as np

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode

# MediaPipe API changed across versions; support both import paths.
try:
    from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat
except Exception:  # pragma: no cover - version compatibility
    from mediapipe import Image as MpImage, ImageFormat

try:
    from mediapipe.tasks.python.vision import drawing_utils
except Exception:  # pragma: no cover - version compatibility
    from mediapipe.python.solutions import drawing_utils


def _models_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
    )


def _pose_model_path() -> str:
    return os.path.join(_models_dir(), "pose_landmarker_lite.task")


class PoseDetector:
    def __init__(self, min_confidence: float = 0.5):
        path = _pose_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pose model missing: {path}\nSee models/README.md for pose_landmarker_lite.task."
            )
        options = pose_landmarker.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=path),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=False,
        )
        self._lm = pose_landmarker.PoseLandmarker.create_from_options(options)
        self._connections = pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS

    def process_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._lm.detect(MpImage(ImageFormat.SRGB, rgb))

    def draw_landmarks(self, frame: np.ndarray, results) -> None:
        if not results.pose_landmarks:
            return
        for pl in results.pose_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                pl,
                self._connections,
                drawing_utils.DrawingSpec(color=(255, 128, 0), thickness=2),
                drawing_utils.DrawingSpec(color=(255, 200, 100), thickness=2),
            )

    def landmarks_to_pixel_dict(self, frame: np.ndarray, results) -> dict[int, tuple[int, int]]:
        if not results.pose_landmarks:
            return {}
        pl = results.pose_landmarks[0]
        h, w = frame.shape[:2]
        out: dict[int, tuple[int, int]] = {}
        for i, lm in enumerate(pl):
            out[i] = (int(lm.x * w), int(lm.y * h))
        return out

    def close(self) -> None:
        self._lm.close()
