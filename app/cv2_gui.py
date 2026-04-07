"""OpenCV highgui — תמיכה גם כשאין GUI (headless)."""
from __future__ import annotations

import cv2


def ensure_opencv_gui_windows() -> None:
    """
    חובה לפרויקט: חלונות מצלמה (cv2.imshow).
    אם הותקן opencv-python-headless (למשל אחרי insightface/albumentations),
    namedWindow נכשל עם 'The function is not implemented'.
    """
    name = "__dronn_cv2_gui_probe__"
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise SystemExit(
            "\n[OpenCV] אין תמיכה בחלונות (HighGUI) — בדרך כלל בגלל "
            "opencv-python-headless.\n\n"
            "  pip uninstall opencv-python-headless -y\n"
            "  pip install --force-reinstall \"opencv-python>=4.8,<5\"\n\n"
            f"מקור: {exc}\n"
        ) from exc
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def destroy_all_windows_safe() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
