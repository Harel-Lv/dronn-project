"""Shared session mode names and in-session hotkey map."""
from __future__ import annotations

MODE_CHOICES = ("gesture", "identity", "tracking", "manual", "webcam_faces", "chase")

MODE_SWITCH_KEYS: dict[int, str] = {
    ord("1"): "manual",
    ord("2"): "gesture",
    ord("3"): "identity",
    ord("4"): "webcam_faces",
    ord("5"): "tracking",
    ord("6"): "chase",
}

MODE_SWITCH_HINT = (
    "1=manual 2=gesture 3=identity 4=webcam_faces 5=tracking 6=chase"
)

# Single OpenCV window reused across in-session mode switches (webcam modes).
SESSION_MAIN_WINDOW = "Dronn"
