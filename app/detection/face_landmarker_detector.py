"""Face mesh via MediaPipe Face Landmarker."""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import face_landmarker
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


def _face_model_path() -> str:
    d = _models_dir()
    primary = os.path.join(d, "face_landmarker.task")
    if os.path.isfile(primary):
        return primary
    pattern = os.path.join(d, "face_landmarker*.task")
    found = sorted(glob.glob(pattern))
    for p in found:
        if os.path.isfile(p):
            return p
    return primary


class FaceLandmarkerDetector:
    def __init__(self, min_confidence: float = 0.5):
        path = _face_model_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Face model missing in {_models_dir()} — expected face_landmarker.task "
                f"(or face_landmarker*.task).\nSee models/README.md."
            )
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=path),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._lm = face_landmarker.FaceLandmarker.create_from_options(options)
        self._connections = face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL

    def process_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._lm.detect(MpImage(ImageFormat.SRGB, rgb))

    def draw_landmarks(self, frame: np.ndarray, results) -> None:
        if not results.face_landmarks:
            return
        for fl in results.face_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                fl,
                self._connections,
                drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1),
                drawing_utils.DrawingSpec(color=(0, 200, 200), thickness=1),
            )

    def first_face_landmarks(self, results):
        if not results.face_landmarks:
            return None
        return results.face_landmarks[0]

    def close(self) -> None:
        self._lm.close()
