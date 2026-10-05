"""HUD: minimal operator overlays (re-exports operator_hud)."""
from __future__ import annotations

from app.ui.operator_hud import (
    draw_battery_badge,
    draw_calibration_overlay,
    draw_face_box,
    draw_operator_hud,
)


def draw_gesture_mode_hud(
    frame,
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
    fpv_stale: bool = False,
    is_airborne: bool = False,
) -> None:
    del dominant_hand, calib_status, calib_ref_px, stab_normal, stab_critical, raw_gesture
    locked = raw_intent == stable_intent
    detail = identity_line
    alert = "FPV STALE" if fpv_stale else None
    draw_operator_hud(
        frame,
        mode=session_mode.upper().replace(" ", "_")[:24],
        command=stable_intent,
        detail=detail,
        alert=alert,
        is_sim=not is_live_drone,
        airborne=is_airborne,
        mode_for_keys=session_mode,
        command_locked=locked,
    )


def draw_manual_mode_hud(
    frame,
    *,
    is_live_tello: bool,
    follow_mode: bool,
    control_mode: str,
    rc: tuple[int, int, int, int],
    last_face: str,
    fpv_stale: bool,
    battery_pct: int | None = None,
) -> None:
    lr, fb, ud, yaw = rc
    detail = f"RC {lr},{fb},{ud},{yaw}"
    if last_face:
        detail += f"  |  {last_face[:20]}"
    if follow_mode:
        detail += "  |  FOLLOW"
    draw_operator_hud(
        frame,
        mode="MANUAL",
        command=control_mode,
        detail=detail,
        alert="FPV STALE" if fpv_stale else None,
        is_sim=not is_live_tello,
        mode_for_keys="manual",
    )
    draw_battery_badge(frame, battery_pct)
