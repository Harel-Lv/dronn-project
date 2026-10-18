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


def is_quit_key(key: int, drone=None) -> bool:
    """Esc always quits. 'q' is yaw-left in manual mode, so it must not end a flight elsewhere."""
    if key == 27:
        return True
    if key == ord("q"):
        if drone is not None and drone.is_airborne:
            print("[session] q ignored while airborne — L = land, Esc = quit (auto-land)")
            return False
        return True
    return False
