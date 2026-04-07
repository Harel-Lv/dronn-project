"""HUD: מצב, כיול, משוב מחווה / פקודה יציבה."""
from __future__ import annotations

import cv2
import numpy as np


def draw_calibration_overlay(
    frame: np.ndarray,
    *,
    seconds_remaining: float,
    total_seconds: float,
    samples_ok: int,
    min_samples: int,
    dominant_hand: str,
) -> None:
    """ספירה לאחור + פס התקדמות בזמן כיול כף."""
    h, w = frame.shape[:2]
    total_seconds = max(total_seconds, 1e-6)
    done = 1.0 - max(0.0, min(1.0, seconds_remaining / total_seconds))
    bar_w = max(40, w - 80)
    bx, bh = 40, 12
    by = max(8, h - 36)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bh), (60, 60, 80), -1)
    fill = int(bar_w * done)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (0, 200, 120), -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bh), (220, 220, 240), 1)

    line1 = (
        f"CALIB: open palm | samples {samples_ok}/{min_samples} | "
        f"hand={dominant_hand} | {seconds_remaining:.1f}s left"
    )
    ty = max(18, min(h - 8, h - 48))
    cv2.putText(
        frame,
        line1[:100],
        (12, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_gesture_mode_hud(
    frame: np.ndarray,
    *,
    session_mode: str,
    dominant_hand: str,
    calib_status: str,
    calib_ref_px: float | None,
    raw_gesture: str,
    raw_intent: str,
    stable_intent: str,
    identity_line: str | None,
    stab_normal: int = 4,
    stab_critical: int = 8,
    is_live_drone: bool = True,
) -> None:
    """
    פס עליון: מצב + יד + כיול.
    בלוק תחתון: מחווה גולמית, פקודה יציבה (ירוק כשננעל), מקשים + מקרא מחוות.
    """
    h, w = frame.shape[:2]

    # --- top banner (semi-transparent strip)
    banner_h = min(120, max(72, h // 3))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 22, 35), -1)
    cv2.addWeighted(
        overlay[:banner_h, :], 0.55, frame[:banner_h, :], 0.45, 0, frame[:banner_h, :]
    )

    if calib_status == "ok" and calib_ref_px is not None:
        calib_txt = f"calib OK ({calib_ref_px:.0f}px)"
        calib_color = (80, 220, 100)
    elif calib_status == "skipped":
        calib_txt = "calib: off"
        calib_color = (180, 180, 200)
    elif calib_status == "cancelled":
        calib_txt = "calib: cancelled — no norm"
        calib_color = (100, 180, 255)
    elif calib_status == "failed":
        calib_txt = "calib FAILED — no norm (see tips below)"
        calib_color = (80, 80, 255)
    else:
        calib_txt = "calib: —"
        calib_color = (200, 200, 220)

    sim = "" if is_live_drone else "[SIMULATION] "
    line_a = f"{sim}MODE: {session_mode}  |  hand: {dominant_hand}  |  {calib_txt}"
    cv2.putText(
        frame,
        line_a[:120],
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        calib_color,
        2,
        cv2.LINE_AA,
    )

    y_sub = 52
    if calib_status == "failed":
        tips = "Tip: bright room, full hand in frame, arm's length from camera"
        cv2.putText(
            frame,
            tips[:95],
            (12, y_sub),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 160, 255),
            2,
            cv2.LINE_AA,
        )
        y_sub += 22
    if identity_line:
        cv2.putText(
            frame,
            identity_line[:100],
            (12, y_sub),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 230, 255),
            2,
            cv2.LINE_AA,
        )
        y_sub += 22

    stab_txt = f"stability: normal={stab_normal} fr | LAND/TAKEOFF={stab_critical} fr"
    cv2.putText(
        frame,
        stab_txt[:100],
        (12, y_sub),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (130, 130, 150),
        1,
        cv2.LINE_AA,
    )

    # --- bottom: gesture feedback (לא לחפוף לפס עליון)
    y_base = max(banner_h + 6, h - 118)
    if y_base + 105 > h:
        y_base = max(banner_h + 4, min(h - 40, h - 105))
    if y_base >= h - 12:
        y_base = max(0, h // 2)
    fo = frame.copy()
    cv2.rectangle(fo, (0, y_base), (w, h), (12, 12, 18), -1)
    cv2.addWeighted(fo[y_base:h, :], 0.5, frame[y_base:h, :], 0.5, 0, frame[y_base:h, :])

    locked = raw_intent == stable_intent
    raw_col = (200, 220, 255)
    stab_col = (80, 255, 120) if locked else (60, 200, 255)

    cv2.putText(
        frame,
        f"detected gesture: {raw_gesture}  ->  intent: {raw_intent}",
        (12, y_base + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        raw_col,
        2,
        cv2.LINE_AA,
    )
    lock_hint = "LOCKED" if locked else "stabilizing..."
    cv2.putText(
        frame,
        f"stable command: {stable_intent}  ({lock_hint})",
        (12, y_base + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        stab_col,
        2,
        cv2.LINE_AA,
    )
    help1 = (
        "Gestures: FIST=land | OPEN=hover | THUMB only (4 fingers curled)=takeoff | "
        "INDEX point=fwd | PEACE=back"
    )
    help2 = "Keys: Esc / q = quit"
    cv2.putText(
        frame,
        help1[:100],
        (12, y_base + 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 175, 190),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        help2[:100],
        (12, y_base + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 175, 190),
        2,
        cv2.LINE_AA,
    )


def draw_manual_mode_hud(
    frame: np.ndarray,
    *,
    is_live_tello: bool,
    follow_mode: bool,
    control_mode: str,
    rc: tuple[int, int, int, int],
    last_face: str,
    fpv_stale: bool,
    battery_pct: int | None = None,
) -> None:
    h, w = frame.shape[:2]
    banner_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 20, 35), -1)
    cv2.addWeighted(overlay[:banner_h, :], 0.52, frame[:banner_h, :], 0.48, 0, frame[:banner_h, :])

    sim = "SIMULATION (no real drone)" if not is_live_tello else "LIVE Tello"
    cv2.putText(
        frame,
        f"MODE: manual  |  {sim}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"CONTROL: {control_mode}",
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 230, 160),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Esc=quit | T=takeoff | L=land | long Tab=FOLLOW | H=hands | I=face-id | N=save",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "WASD=move | R/F=up/down | Q/E=yaw",
        (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (160, 200, 230),
        2,
        cv2.LINE_AA,
    )
    _draw_battery(frame, battery_pct)

    lr, fb, ud, yaw = rc
    warn = " | FPV STALE — RC zeroed" if fpv_stale else ""
    line = (
        f"RC lr={lr} fb={fb} ud={ud} yaw={yaw} | FOLLOW={follow_mode} | {last_face}{warn}"
    )
    cv2.putText(
        frame,
        line[:115],
        (10, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255) if not fpv_stale else (100, 100, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_battery(frame: np.ndarray, battery_pct: int | None) -> None:
    """Simple battery icon + percent on top-right of manual HUD."""
    if battery_pct is None:
        return
    b = max(0, min(100, int(battery_pct)))
    h, w = frame.shape[:2]
    x2 = w - 16
    y1 = 12
    body_w = 64
    body_h = 22
    x1 = x2 - body_w
    y2 = y1 + body_h
    # Body
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)
    # Tip
    cv2.rectangle(frame, (x2 + 1, y1 + 6), (x2 + 6, y2 - 6), (220, 220, 220), -1)
    # Fill
    fill_w = int((body_w - 4) * (b / 100.0))
    color = (0, 220, 0) if b >= 30 else (0, 180, 255) if b >= 15 else (0, 0, 255)
    cv2.rectangle(frame, (x1 + 2, y1 + 2), (x1 + 2 + fill_w, y2 - 2), color, -1)
    cv2.putText(
        frame,
        f"{b}%",
        (x1 - 54, y2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
