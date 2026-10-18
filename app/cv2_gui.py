"""OpenCV highgui — fail clearly when there is no GUI support (headless build)."""
from __future__ import annotations

import cv2


def ensure_opencv_gui_windows() -> None:
    """
    Required by the project: camera windows (cv2.imshow).
    If opencv-python-headless got installed (e.g. pulled in by insightface/albumentations),
    namedWindow fails with 'The function is not implemented'.
    """
    name = "__dronn_cv2_gui_probe__"
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise SystemExit(
            "\n[OpenCV] no window support (HighGUI) — usually caused by "
            "opencv-python-headless.\n\n"
            "  pip uninstall opencv-python-headless -y\n"
            "  pip install --force-reinstall \"opencv-python>=4.8,<5\"\n\n"
            f"Source: {exc}\n"
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
