"""Minimal English operator HUD — fast cv2.putText, no Pillow."""
from __future__ import annotations

import cv2
import numpy as np

_KEYS = "L=land  q=quit"
_SWITCH = "1-6=mode  q=quit"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TOP_H = 76
_BOT_H = 40
_BOT_H_2 = 58
_MAX_KEY_CHARS = 118

# (primary controls, secondary / common)
_MODE_KEY_LINES: dict[str, tuple[str, str]] = {
    "manual": (
        "WASD=move  R/F=up-down  Q/E=yaw  Tab=FOLLOW  H=hands",
        f"T=takeoff  L=land  {_SWITCH}",
    ),
    "gesture": (
        "FIST=land  OPEN=hover  THUMB=takeoff  POINT=fwd  PEACE=back",
        f"L=emergency land  {_SWITCH}",
    ),
    "webcam_faces": (
        "gestures (above) + enrolled face required",
        f"L=emergency land  {_SWITCH}",
    ),
    "identity": (
        "green = match  |  orange = unknown",
        f"N=enroll on screen  {_SWITCH}",
    ),
    "tracking": (
        "auto-follow locked person  (calibrate on entry)",
        f"T=takeoff  L=land  {_SWITCH}",
    ),
    "chase": (
        "largest face = red box  |  hold SPACE to rush",
        f"T=takeoff  L=land  {_SWITCH}",
    ),
}

_CALIB_KEY_LINES: dict[str, tuple[str, str]] = {
    "gesture": ("CALIB: hold OPEN PALM at arm's length", f"Esc=cancel  {_SWITCH}"),
    "webcam_faces": ("CALIB: hold OPEN PALM at arm's length", f"Esc=cancel  {_SWITCH}"),
    "tracking": ("CALIB: stand in frame — locking your body", f"L=land  {_SWITCH}"),
}


def normalize_mode_key(session_mode: str) -> str:
    s = str(session_mode).lower().replace(" ", "_")
    if "webcam" in s or "known_person" in s or "gate" in s:
        return "webcam_faces"
    if "follow" in s or "tracking" in s or "body" in s:
        return "tracking"
    if "chase" in s:
        return "chase"
    if "identity" in s:
        return "identity"
    if "manual" in s:
        return "manual"
    return "gesture"


def mode_key_lines(mode: str, *, calibrating: bool = False) -> tuple[str, str]:
    key = normalize_mode_key(mode)
    if calibrating:
        calib = _CALIB_KEY_LINES.get(key)
        if calib is not None:
            return calib
    return _MODE_KEY_LINES.get(
        key,
        (f"L=land  {_SWITCH}", ""),
    )


def _top_bar(frame: np.ndarray, height: int | None = None) -> None:
    h, w = frame.shape[:2]
    bar_h = min(height or _TOP_H, h)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (18, 20, 28), -1)


def _bottom_bar(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    y0 = max(0, h - _BOT_H)
    cv2.rectangle(frame, (0, y0), (w, h), (18, 20, 28), -1)


def _text(
    frame: np.ndarray,
    text: str,
    y: int,
    *,
    scale: float = 0.58,
    color: tuple[int, int, int] = (235, 235, 240),
    thickness: int = 2,
) -> None:
    if not text:
        return
    cv2.putText(
        frame,
        text[:96],
        (12, y),
        _FONT,
        scale,
        color,
        thickness,
        cv2.LINE_8,
    )


def draw_operator_hud(
    frame: np.ndarray,
    *,
    mode: str,
    command: str,
    detail: str | None = None,
    alert: str | None = None,
    is_sim: bool = False,
    airborne: bool = False,
    keys_line: str | None = None,
    keys_line2: str | None = None,
    command_locked: bool | None = None,
    mode_for_keys: str | None = None,
    calibrating: bool = False,
) -> None:
    """
    Three-line operator strip (top) + keys (bottom).
    mode: short mode name, command: primary action the pilot cares about.
    Pass mode_for_keys to auto-fill keys_line / keys_line2 from mode_key_lines().
    """
    if mode_for_keys is not None and keys_line is None:
        keys_line, keys_line2 = mode_key_lines(mode_for_keys, calibrating=calibrating)
    keys_line = keys_line or _KEYS
    bar_h = 76 if detail else 58
    _top_bar(frame, bar_h)
    tag = "SIM" if is_sim else "LIVE"
    air = " | AIR" if airborne and not is_sim else ""
    alert_s = f" | {alert}" if alert else ""
    _text(frame, f"{tag} | {mode}{air}{alert_s}", 22, scale=0.62)

    if command_locked is True:
        cmd_color = (80, 255, 120)
        suffix = ""
    elif command_locked is False:
        cmd_color = (80, 200, 255)
        suffix = " ..."
    else:
        cmd_color = (220, 220, 230)
        suffix = ""
    _text(frame, f"CMD: {command}{suffix}", 44, scale=0.68, color=cmd_color, thickness=2)

    if detail:
        _text(frame, detail, 64, scale=0.5, color=(150, 160, 180), thickness=1)

    h, w = frame.shape[:2]
    bot_h = _BOT_H_2 if keys_line2 else _BOT_H
    y0 = max(0, h - bot_h)
    cv2.rectangle(frame, (0, y0), (w, h), (18, 20, 28), -1)
    if keys_line2:
        _text(frame, keys_line[:_MAX_KEY_CHARS], h - 36, scale=0.48, color=(170, 175, 190), thickness=1)
        _text(frame, keys_line2[:_MAX_KEY_CHARS], h - 14, scale=0.48, color=(150, 155, 170), thickness=1)
    else:
        _text(frame, keys_line[:_MAX_KEY_CHARS], h - 14, scale=0.5, color=(170, 175, 190), thickness=1)


def draw_battery_badge(frame: np.ndarray, battery_pct: int | None) -> None:
    if battery_pct is None:
        return
    b = max(0, min(100, int(battery_pct)))
    h, w = frame.shape[:2]
    color = (0, 220, 0) if b >= 30 else (0, 180, 255) if b >= 15 else (0, 0, 255)
    cv2.putText(
        frame,
        f"BAT {b}%",
        (w - 108, 26),
        _FONT,
        0.58,
        color,
        2,
        cv2.LINE_8,
    )


def draw_calibration_overlay(
    frame: np.ndarray,
    *,
    seconds_remaining: float,
    samples_ok: int,
    min_samples: int,
    mode_for_keys: str = "gesture",
) -> None:
    h, w = frame.shape[:2]
    k1, k2 = mode_key_lines(mode_for_keys, calibrating=True)
    bot_h = _BOT_H_2 if k2 else _BOT_H
    y0 = max(0, h - bot_h - 36)
    cv2.rectangle(frame, (0, y0), (w, h - 36), (18, 20, 28), -1)
    if k2:
        _text(frame, k1[:_MAX_KEY_CHARS], h - 66, scale=0.48, color=(170, 175, 190), thickness=1)
        _text(frame, k2[:_MAX_KEY_CHARS], h - 48, scale=0.48, color=(150, 155, 170), thickness=1)
    else:
        _text(frame, k1[:_MAX_KEY_CHARS], h - 48, scale=0.5, color=(170, 175, 190), thickness=1)
    total = max(seconds_remaining, 0.1) + 1.0
    done = 1.0 - max(0.0, min(1.0, seconds_remaining / max(total, 1.0)))
    bar_w = max(40, w - 80)
    bx, by, bh = 40, h - 28, 10
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bh), (50, 50, 65), -1)
    fill = int(bar_w * done)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (0, 200, 120), -1)
    _text(
        frame,
        f"CALIB open palm  {samples_ok}/{min_samples}  {seconds_remaining:.1f}s",
        h - 44,
        scale=0.55,
        color=(0, 255, 255),
    )


def draw_face_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    label: str | None = None,
    thickness: int = 2,
) -> None:
    x0, y0, x1, y1 = bbox
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
    if label:
        _text(frame, label, max(18, y0 - 8), scale=0.55, color=color, thickness=2)
