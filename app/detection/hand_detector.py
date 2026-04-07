from __future__ import annotations

import os

import cv2
import numpy as np

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
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
try:
    from mediapipe.framework.formats import landmark_pb2
except Exception:  # pragma: no cover - version compatibility
    landmark_pb2 = None


def _get_model_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "models", "hand_landmarker.task")


class HandDetector:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        model_path = _get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "See models/README.md"
            )

        options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = hand_landmarker.HandLandmarker.create_from_options(options)
        self._connections = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS

    def process_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = MpImage(ImageFormat.SRGB, rgb)
        return self._landmarker.detect(mp_img)

    def draw_landmarks(self, frame: np.ndarray, results) -> None:
        if not results.hand_landmarks:
            return
        for hand_landmarks in results.hand_landmarks:
            # MediaPipe versions differ: some return a proto with `.landmark`,
            # others return a plain list of normalized landmarks.
            try:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._connections,
                )
                continue
            except Exception:
                pass

            if landmark_pb2 is None or not isinstance(hand_landmarks, list):
                continue
            try:
                proto = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(
                            x=float(getattr(lm, "x", 0.0)),
                            y=float(getattr(lm, "y", 0.0)),
                            z=float(getattr(lm, "z", 0.0)),
                        )
                        for lm in hand_landmarks
                    ]
                )
                drawing_utils.draw_landmarks(
                    frame,
                    proto,
                    self._connections,
                )
            except Exception:
                continue

    def close(self) -> None:
        self._landmarker.close()

    def extract_landmarks(self, frame: np.ndarray, results):
        if not results.hand_landmarks:
            return []
        if frame is None or len(frame.shape) < 2:
            return []
        frame_height, frame_width, _ = frame.shape
        all_hands = []
        for hand_landmarks in results.hand_landmarks:
            lm_dict = {}
            for landmark_id, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                lm_dict[landmark_id] = (x, y)
            all_hands.append(lm_dict)
        return all_hands

    def extract_hands_with_handedness(
        self, frame: np.ndarray, results
    ) -> list[tuple[dict, str]]:
        """
        מחזיר רשימת (מילון נקודות, תווית יד).
        תווית: בדרך כלל 'Left' או 'Right' מתוך MediaPipe handedness.
        אם אין handedness — מחזיר 'Unknown' (הבחירה הדומיננטית תיפול ליד הראשונה).
        """
        if not results.hand_landmarks:
            return []
        if frame is None or len(frame.shape) < 2:
            return []
        fh, fw, _ = frame.shape
        handedness_lists = getattr(results, "handedness", None) or []

        out: list[tuple[dict, str]] = []
        for hi, hand_landmarks in enumerate(results.hand_landmarks):
            lm_dict: dict[int, tuple[int, int]] = {}
            for landmark_id, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * fw)
                y = int(landmark.y * fh)
                lm_dict[landmark_id] = (x, y)

            label = "Unknown"
            if hi < len(handedness_lists):
                cats = handedness_lists[hi]
                if cats:
                    cn = getattr(cats[0], "category_name", None)
                    if cn:
                        label = str(cn)
            out.append((lm_dict, label))
        return out
