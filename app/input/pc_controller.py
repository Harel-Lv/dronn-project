"""Keyboard (pynput) state for manual PC control mode."""
from __future__ import annotations

import sys
import time
from typing import Callable

from pynput import keyboard

# מקשים לפי מיקום פיזי (Windows VK) — עובד גם כשהמקלדת בעברית
_VK = {
    "w": 0x57,
    "s": 0x53,
    "a": 0x41,
    "d": 0x44,
    "r": 0x52,
    "f": 0x46,
    "q": 0x51,
    "e": 0x45,
    "l": 0x4C,
    "t": 0x54,  # takeoff (לא נכנס ל-compute_manual_rc)
    "h": 0x48,  # hand mode toggle
    "i": 0x49,  # face-id toggle
    "n": 0x4E,  # save unknown
}


class PcInputController:
    """Tab long = FOLLOW, T = takeoff once, L = emergency land."""

    TAB_HOLD_SEC = 0.55

    def __init__(self) -> None:
        self._down: set[keyboard.Key | keyboard.KeyCode] = set()
        self._listener: keyboard.Listener | None = None
        self._tab_down_at: float | None = None
        self.follow_mode: bool = False
        self.quit_requested: bool = False
        self.takeoff_requested: bool = False
        self.emergency_land: bool = False
        self.hand_mode_toggle_requested: bool = False
        self.face_id_toggle_requested: bool = False
        self.face_save_requested: bool = False
        self._on_follow_toggle: Callable[[bool], None] | None = None

    def set_follow_toggle_callback(self, fn: Callable[[bool], None] | None) -> None:
        self._on_follow_toggle = fn

    def start(self) -> None:
        if self._listener is not None:
            return

        def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if key is None:
                return
            # Prevent auto-repeat from causing multiple toggles.
            if key in self._down:
                return
            self._down.add(key)
            if key == keyboard.Key.esc:
                self.quit_requested = True
            try:
                if self._char_or_vk(key, "t", _VK["t"]):
                    self.takeoff_requested = True
                if self._char_or_vk(key, "l", _VK["l"]):
                    self.emergency_land = True
                if self._char_or_vk(key, "h", _VK["h"]):
                    self.hand_mode_toggle_requested = True
                if self._char_or_vk(key, "i", _VK["i"]):
                    self.face_id_toggle_requested = True
                if self._char_or_vk(key, "n", _VK["n"]):
                    self.face_save_requested = True
            except (AttributeError, TypeError):
                pass
            if key == keyboard.Key.tab:
                self._tab_down_at = time.monotonic()

        def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if key is None:
                return
            self._down.discard(key)
            if key == keyboard.Key.tab and self._tab_down_at is not None:
                held = time.monotonic() - self._tab_down_at
                if held >= self.TAB_HOLD_SEC:
                    self.follow_mode = not self.follow_mode
                    if self._on_follow_toggle:
                        self._on_follow_toggle(self.follow_mode)
                    print(f"[manual] FOLLOW mode = {self.follow_mode}")
                self._tab_down_at = None

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
        except OSError as exc:
            hint = ""
            if sys.platform == "win32":
                hint = " ב-Windows נסה להריץ את PowerShell/Cursor כמנהל (Run as administrator)."
            raise RuntimeError(
                "pynput לא הצליח להאזין למקלדת (נדרש למצב manual)."
                f"{hint} פרטים: {exc}"
            ) from exc

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._down.clear()
        self._tab_down_at = None
        self.takeoff_requested = False
        self.emergency_land = False
        self.hand_mode_toggle_requested = False
        self.face_id_toggle_requested = False
        self.face_save_requested = False

    @staticmethod
    def _char_or_vk(key: keyboard.Key | keyboard.KeyCode, char: str, vk: int) -> bool:
        try:
            if hasattr(key, "char") and key.char == char:
                return True
        except (AttributeError, TypeError):
            pass
        try:
            if hasattr(key, "vk") and key.vk is not None and int(key.vk) == vk:
                return True
        except (AttributeError, TypeError, ValueError):
            pass
        return False

    def _is(self, *chars: str) -> bool:
        for c in chars:
            vk = _VK.get(c)
            if vk is None:
                continue
            for k in self._down:
                if self._char_or_vk(k, c, vk):
                    return True
        return False

    def compute_manual_rc(self, speed: int = 55) -> tuple[int, int, int, int]:
        """W/S forward-back, A/D left-right, R/F up-down, Q/E yaw (Tello RC axes)."""
        lr = fb = ud = yaw = 0
        s = max(10, min(100, int(speed)))
        if self._is("w"):
            fb = s
        if self._is("s"):
            fb = -s
        if self._is("a"):
            lr = -s
        if self._is("d"):
            lr = s
        if self._is("r"):
            ud = s
        if self._is("f"):
            ud = -s
        if self._is("q"):
            yaw = -s
        if self._is("e"):
            yaw = s
        return lr, fb, ud, yaw
