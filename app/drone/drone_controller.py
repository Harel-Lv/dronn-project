"""Simulated or DJI Tello drone backend."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.drone.safety_rc import clamp_rc


class DroneController:
    def __init__(self, config: dict) -> None:
        self._cfg = config.get("drone", {})
        self._enabled = bool(self._cfg.get("enabled", False))
        self._backend = str(self._cfg.get("backend", "simulated")).lower()
        self._tello: Any = None
        self._frame_reader: Any = None
        self._last_rc = (0, 0, 0, 0)
        self._connected = False
        self._last_battery: int | None = None
        self._last_battery_at: float = 0.0
        self._battery_display: float | None = None
        # Exponential moving average smoothing for HUD stability.
        self._battery_ema_alpha: float = 0.25

    @property
    def is_live_tello(self) -> bool:
        return self._enabled and self._backend == "tello"

    def connect(self) -> None:
        if not self._enabled:
            print("[drone] simulation — לא מחובר לרחפן אמיתי")
            self._connected = True
            return
        if self._backend != "tello":
            print(f"[drone] backend={self._backend!r} — ללא חיבור רשת")
            self._connected = True
            return

        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError(
                "נדרש djitellopy ל-Tello: pip install djitellopy"
            ) from exc

        retries = max(1, int(self._cfg.get("tello_connect_retries", 3)))
        delay = max(0.1, float(self._cfg.get("tello_connect_retry_delay_sec", 1.5)))
        # tello_host: null ב-YAML → None; אסור להעביר host=None ל-djitellopy (שובר sendto).
        host_raw = self._cfg.get("tello_host")
        last_err: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                if host_raw is not None and str(host_raw).strip():
                    self._tello = Tello(host=str(host_raw).strip())
                else:
                    self._tello = Tello()
                self._battery_display = None
                self._tello.connect()
                b = int(self._tello.get_battery())
                self._last_battery = b
                self._last_battery_at = time.monotonic()
                print(f"[drone] Tello: battery={b}%")
                if bool(self._cfg.get("show_tello_fpv", False)):
                    self._tello.streamon()
                    self._frame_reader = self._tello.get_frame_read()
                self._connected = True
                return
            except Exception as exc:
                last_err = exc
                self._tello = None
                self._frame_reader = None
                if attempt < retries:
                    print(
                        f"[drone] חיבור נכשל ({attempt}/{retries}): {exc} — "
                        f"מנסה שוב בעוד {delay:.1f}s…"
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"Tello: חיבור נכשל אחרי {retries} ניסיונות. אחרון: {last_err}"
        ) from last_err

    def disconnect(self) -> None:
        if self._tello is not None:
            try:
                self.send_rc(0, 0, 0, 0)
            except Exception:
                pass
            try:
                if bool(self._cfg.get("show_tello_fpv", False)):
                    self._tello.streamoff()
            except Exception:
                pass
            try:
                self._tello.end()
            except Exception:
                pass
        self._tello = None
        self._frame_reader = None
        self._connected = False
        self._battery_display = None

    def takeoff(self) -> None:
        if self._tello is None:
            print("[drone:sim] TAKEOFF")
            return
        self._tello.takeoff()

    def land(self) -> None:
        if self._tello is None:
            print("[drone:sim] LAND")
            return
        self._tello.land()

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        lr, fb, ud, yaw = clamp_rc(lr, fb, ud, yaw)
        self._last_rc = (lr, fb, ud, yaw)
        if self._tello is None:
            if any(self._last_rc):
                print(f"[drone:sim] RC {self._last_rc}")
            return
        self._tello.send_rc_control(lr, fb, ud, yaw)

    def read_fpv_frame(self) -> np.ndarray | None:
        if self._frame_reader is None:
            return None
        try:
            f = self._frame_reader.frame
        except Exception:
            return None
        if f is None or f.size == 0:
            return None
        return f

    @property
    def last_rc(self) -> tuple[int, int, int, int]:
        return self._last_rc

    def get_battery_percent(self, *, min_interval_sec: float = 1.0) -> int | None:
        """
        Battery percent for HUD.
        Uses cached value for up to min_interval_sec to avoid chatty SDK calls.
        """
        if self._tello is None:
            return self._last_battery
        now = time.monotonic()
        if (
            self._last_battery is not None
            and (now - self._last_battery_at) < float(min_interval_sec)
        ):
            return self._last_battery
        try:
            raw = int(self._tello.get_battery())
            # Smooth UI value to reduce short-term voltage/load dips.
            # Also prevent "increasing" the displayed battery due to noise.
            if self._battery_display is None:
                self._battery_display = float(raw)
            else:
                ema = self._battery_ema_alpha * float(raw) + (
                    1.0 - self._battery_ema_alpha
                ) * float(self._battery_display)
                # Clamp so the HUD doesn't jump up after a temporary dip.
                self._battery_display = float(min(self._battery_display, ema))

            b = int(round(self._battery_display))
            self._last_battery = b
            self._last_battery_at = now
            return b
        except Exception:
            return self._last_battery
