"""Smoothing for auto-follow RC: damp oscillation without delaying safety stops."""
from __future__ import annotations

import time


class RcSmoother:
    """
    Exponential smoothing + per-update rate limit on each RC axis.
    An all-zero target bypasses smoothing: stops (target lost, stale video, too close)
    must take effect immediately.
    """

    def __init__(self, *, alpha: float = 0.4, max_step: int = 15) -> None:
        self._alpha = min(1.0, max(0.05, float(alpha)))
        self._max_step = max(1, int(max_step))
        self._value = [0.0, 0.0, 0.0, 0.0]

    def reset(self) -> None:
        self._value = [0.0, 0.0, 0.0, 0.0]

    def update(self, target: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if not any(target):
            self.reset()
            return 0, 0, 0, 0
        out: list[int] = []
        for i, t in enumerate(target):
            prev = self._value[i]
            step = self._alpha * (float(t) - prev)
            step = max(-self._max_step, min(self._max_step, step))
            self._value[i] = prev + step
            out.append(int(round(self._value[i])))
        return out[0], out[1], out[2], out[3]


class ErrorRate:
    """Smoothed d(error)/dt for the derivative (damping) term."""

    def __init__(self, *, alpha: float = 0.5, max_gap_sec: float = 0.5) -> None:
        self._alpha = min(1.0, max(0.05, float(alpha)))
        self._max_gap = float(max_gap_sec)
        self._last_err: float | None = None
        self._last_t = 0.0
        self.rate = 0.0

    def reset(self) -> None:
        self._last_err = None
        self.rate = 0.0

    def update(self, err: float, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if self._last_err is not None:
            dt = now - self._last_t
            if 1e-3 < dt <= self._max_gap:
                raw = (err - self._last_err) / dt
                self.rate += self._alpha * (raw - self.rate)
            elif dt > self._max_gap:
                # Old sample: a derivative across the gap would be a spike.
                self.rate = 0.0
        self._last_err = err
        self._last_t = now
        return self.rate
