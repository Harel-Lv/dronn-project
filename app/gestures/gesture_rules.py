"""Gesture labels → high-level drone intents (applied after stabilizer)."""
from __future__ import annotations

# Intents consumed by run_gesture_session
HOVER = "HOVER"
TAKEOFF = "TAKEOFF"
LAND = "LAND"
FORWARD = "FORWARD"
BACK = "BACK"
LEFT = "LEFT"
RIGHT = "RIGHT"
UP = "UP"
DOWN = "DOWN"
ROTATE_CW = "ROTATE_CW"
ROTATE_CCW = "ROTATE_CCW"
NO_GESTURE = "NO_GESTURE"


def gesture_to_drone_intent(gesture: str) -> str:
    g = str(gesture).upper()
    mapping = {
        "NONE": NO_GESTURE,
        "UNKNOWN": NO_GESTURE,
        "FIST": LAND,
        "OPEN_PALM": HOVER,
        "THUMB_UP": TAKEOFF,
        "POINT": FORWARD,
        "PEACE": BACK,
    }
    return mapping.get(g, NO_GESTURE)
