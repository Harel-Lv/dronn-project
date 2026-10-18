"""MediaPipe VIDEO running mode: tracks across frames (faster, steadier than IMAGE)."""
from __future__ import annotations

import time

from mediapipe.tasks.python.vision.core import vision_task_running_mode

RunningMode = vision_task_running_mode.VisionTaskRunningMode


def running_mode(video: bool):
    return RunningMode.VIDEO if video else RunningMode.IMAGE


class VideoTimestamps:
    """detect_for_video() rejects a timestamp that isn't strictly greater than the last one."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._last = -1

    def next_ms(self) -> int:
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        return ts
