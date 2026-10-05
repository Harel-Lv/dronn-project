"""Shared drone/webcam for seamless in-session mode switches."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2

from app.camera.webcam import Webcam
from app.cv2_gui import destroy_all_windows_safe
from app.drone.drone_controller import DroneController
from app.session_modes import SESSION_MAIN_WINDOW
from app.ui.operator_hud import draw_operator_hud


@dataclass
class SessionContext:
    opts: object
    base_config: dict
    drone: DroneController | None = None
    webcam: Webcam | None = None
    _drone_connected: bool = field(default=False, repr=False)
    hover_before_switch_sec: float = 0.35
    preserve_main_window: bool = False
    gesture_calib: tuple[float | None, dict[str, Any]] | None = None
    tracking_calib: tuple[float, float, float] | None = None  # baseline, size_min, size_max

    def ensure_webcam(self, camera_index: int) -> Webcam:
        if self.webcam is None:
            self.webcam = Webcam(camera_index=camera_index)
            self.webcam.open()
            print("WEBCAM OPENED")
        return self.webcam

    def ensure_drone(self, config: dict) -> DroneController:
        if self.drone is None:
            self.drone = DroneController(config)
        if not self._drone_connected:
            self.drone.connect()
            self._drone_connected = True
            dc = config.get("drone", {})
            backend = str(dc.get("backend", "")).lower()
            rt_on = bool(config.get("rt_control", {}).get("enabled", False))
            if dc.get("enabled") and backend == "tello":
                print("DRONE: Tello backend enabled - ensure Wi-Fi connected to drone")
            elif dc.get("enabled") and backend == "rt_cpp" and rt_on:
                print("DRONE: rt_cpp — ensure dronn_rt_service is running (scripts/start_rt_service.py)")
            else:
                print("DRONE: simulation (use --tello or config.yaml for real Tello)")
        return self.drone

    def prepare_mode_switch(self) -> None:
        """Zero RC; brief hover pause when airborne before loading next mode."""
        if self.drone is None:
            return
        try:
            self.drone.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        if self.drone.is_airborne:
            sec = max(0.0, float(self.hover_before_switch_sec))
            print(f"[session] airborne — hover {sec:.1f}s before mode switch")
            if sec > 0:
                time.sleep(sec)

    def finish_mode_window(self, window: str, requested_mode: str | None) -> None:
        """Keep main window open when switching modes; destroy only on quit."""
        if requested_mode:
            self.preserve_main_window = True
            return
        if window == SESSION_MAIN_WINDOW and self.preserve_main_window:
            return
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass

    def show_mode_switch_overlay(
        self,
        next_mode: str,
        *,
        window: str = SESSION_MAIN_WINDOW,
        seconds: float = 0.45,
    ) -> None:
        if self.webcam is None:
            return
        label = str(next_mode).upper().replace("_", " ")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        deadline = time.monotonic() + max(0.1, seconds)
        while time.monotonic() < deadline:
            frame = self.webcam.read_frame()
            if frame is not None:
                draw_operator_hud(
                    frame,
                    mode="SWITCHING",
                    command=label,
                    keys_line="keep camera open — loading mode",
                )
                cv2.imshow(window, frame)
            if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                break

    def shutdown(self) -> None:
        if self.webcam is not None:
            try:
                self.webcam.release()
            except Exception:
                pass
            self.webcam = None
        if self.drone is not None:
            try:
                self.drone.disconnect()
            except Exception:
                pass
            self.drone = None
            self._drone_connected = False
        destroy_all_windows_safe()
