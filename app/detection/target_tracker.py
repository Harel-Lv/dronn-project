"""Target tracking preview — face Haar cascade, alignment hints (simulation only)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
MOVE_FORWARD = "MOVE_FORWARD"
MOVE_BACKWARD = "MOVE_BACKWARD"
HOLD = "HOLD"
NO_TARGET = "NO_TARGET"

CENTER_DEADZONE = 0.15
SIZE_MIN_RATIO = 0.03
SIZE_MAX_RATIO = 0.25


def get_target_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def get_frame_center(frame: np.ndarray) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    return (w // 2, h // 2)


def compute_alignment_command(
    frame: np.ndarray,
    target_bbox: Tuple[int, int, int, int] | None,
    center_deadzone: float = CENTER_DEADZONE,
    size_min_ratio: float = SIZE_MIN_RATIO,
    size_max_ratio: float = SIZE_MAX_RATIO,
) -> str:
    if target_bbox is None:
        return NO_TARGET

    frame_h, frame_w = frame.shape[:2]
    frame_center = get_frame_center(frame)
    target_center = get_target_center(target_bbox)

    x, y, w, h = target_bbox
    target_area = w * h
    frame_area = frame_w * frame_h
    size_ratio = target_area / frame_area if frame_area > 0 else 0

    center_x, center_y = frame_center
    tgt_x, tgt_y = target_center
    deadzone_x = int(frame_w * center_deadzone)

    if tgt_x < center_x - deadzone_x:
        return TURN_LEFT
    if tgt_x > center_x + deadzone_x:
        return TURN_RIGHT

    if size_ratio < size_min_ratio:
        return MOVE_FORWARD
    if size_ratio > size_max_ratio:
        return MOVE_BACKWARD

    return HOLD


def bbox_area_ratio(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return 0.0
    _, _, w, h = bbox
    return (w * h) / frame_area


def detect_target(frame: np.ndarray, face_cascade: cv2.CascadeClassifier) -> Tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
    except cv2.error:
        return None
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


class TargetTracker:
    def __init__(
        self,
        center_deadzone: float = CENTER_DEADZONE,
        size_min_ratio: float = SIZE_MIN_RATIO,
        size_max_ratio: float = SIZE_MAX_RATIO,
    ) -> None:
        self.center_deadzone = center_deadzone
        self.size_min_ratio = size_min_ratio
        self.size_max_ratio = size_max_ratio
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def apply_calibration(
        self,
        *,
        baseline_ratio: float,
        small_factor: float,
        large_factor: float,
    ) -> None:
        self.size_min_ratio = max(0.001, baseline_ratio * small_factor)
        self.size_max_ratio = min(0.95, baseline_ratio * large_factor)

    def process_frame(self, frame: np.ndarray) -> tuple[Tuple[int, int, int, int] | None, str]:
        bbox = detect_target(frame, self.face_cascade)
        cmd = compute_alignment_command(
            frame,
            bbox,
            self.center_deadzone,
            self.size_min_ratio,
            self.size_max_ratio,
        )
        return bbox, cmd
