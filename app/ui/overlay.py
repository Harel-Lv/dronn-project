"""OpenCV drawing helpers for tracking / debug HUD."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_tracking_overlay(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int] | None,
    target_center: Tuple[int, int] | None,
    frame_center: Tuple[int, int],
    command: str,
) -> None:
    fh, fw = frame.shape[:2]
    cx, cy = frame_center
    cv2.drawMarker(
        frame,
        (cx, cy),
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    if target_center is not None:
        tx, ty = target_center
        cv2.circle(frame, (tx, ty), 8, (0, 128, 255), -1)

    cv2.putText(
        frame,
        f"Cmd: {command}",
        (16, fh - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
