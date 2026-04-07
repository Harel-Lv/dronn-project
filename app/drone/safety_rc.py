"""Clamp / sanitize RC values for Tello send_rc_control."""
from __future__ import annotations


def clamp_rc(
    left_right: int,
    forward_back: int,
    up_down: int,
    yaw: int,
    *,
    lo: int = -100,
    hi: int = 100,
) -> tuple[int, int, int, int]:
    def _c(x: int) -> int:
        return max(lo, min(hi, int(x)))

    return _c(left_right), _c(forward_back), _c(up_down), _c(yaw)
