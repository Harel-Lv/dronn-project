"""Hold-to-chase: detect any face (red box), Space hold = max-speed RC toward target."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from pynput import keyboard

from app.drone.drone_controller import DroneController
from app.identity.identity_perf import resize_frame_max_width, scale_bbox
from app.session_modes import MODE_SWITCH_HINT, MODE_SWITCH_KEYS, SESSION_MAIN_WINDOW
from app.ui.operator_hud import draw_operator_hud

_CHASE_WINDOW = "Chase — Tello FPV"
_SIM_WINDOW = SESSION_MAIN_WINDOW
_VK_SPACE = 0x20
_RED_BGR = (0, 0, 255)


@dataclass
class FaceTarget:
    cx: int
    cy: int
    bbox: tuple[int, int, int, int]
    size_ratio: float


class _SpaceHoldController:
    """Space must be held for hold_seconds before chase RC is enabled."""

    def __init__(self, hold_seconds: float = 0.35) -> None:
        self._hold_sec = max(0.05, float(hold_seconds))
        self._down: set[keyboard.Key | keyboard.KeyCode] = set()
        self._space_since: float | None = None
        self._listener: keyboard.Listener | None = None
        self.chase_armed = False

    @staticmethod
    def _is_space(key: keyboard.Key | keyboard.KeyCode) -> bool:
        if key == keyboard.Key.space:
            return True
        try:
            if hasattr(key, "char") and key.char == " ":
                return True
        except (AttributeError, TypeError):
            pass
        try:
            if hasattr(key, "vk") and key.vk is not None and int(key.vk) == _VK_SPACE:
                return True
        except (AttributeError, TypeError, ValueError):
            pass
        return False

    def start(self) -> None:
        if self._listener is not None:
            return

        def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if key is None:
                return
            if key in self._down:
                return
            self._down.add(key)
            if self._is_space(key) and self._space_since is None:
                self._space_since = time.monotonic()

        def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if key is None:
                return
            self._down.discard(key)
            if self._is_space(key):
                self._space_since = None
                self.chase_armed = False

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
        except OSError as exc:
            hint = ""
            if sys.platform == "win32":
                hint = " נסה להריץ PowerShell כמנהל."
            raise RuntimeError(f"pynput לא זמין למצב chase.{hint} {exc}") from exc

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._down.clear()
        self._space_since = None
        self.chase_armed = False

    def update(self) -> bool:
        """Return True when Space held long enough to chase."""
        if self._space_since is None:
            self.chase_armed = False
            return False
        armed = (time.monotonic() - self._space_since) >= self._hold_sec
        self.chase_armed = armed
        return armed


def _bbox_to_target(
    bbox: np.ndarray | tuple[int, ...], frame_w: int, frame_h: int
) -> FaceTarget | None:
    if bbox is None or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if x1 <= x0 or y1 <= y0:
        return None
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    area = max(1, (x1 - x0) * (y1 - y0))
    size_ratio = area / max(1, frame_w * frame_h)
    return FaceTarget(cx=cx, cy=cy, bbox=(x0, y0, x1, y1), size_ratio=size_ratio)


def _detect_largest_face(insight, frame: np.ndarray, max_width: int) -> FaceTarget | None:
    small, inv_scale = resize_frame_max_width(frame, max_width)
    _emb, bbox = insight.embed(small)
    if bbox is None:
        return None
    scaled = scale_bbox(bbox, inv_scale)
    if scaled is None:
        return None
    fh, fw = frame.shape[:2]
    return _bbox_to_target(scaled, fw, fh)


def _compute_chase_rc(
    target: FaceTarget,
    frame_w: int,
    *,
    max_speed: int,
    stop_size_ratio: float,
) -> tuple[int, int, int, int]:
    if target.size_ratio >= stop_size_ratio:
        return 0, 0, 0, 0
    err_x = (target.cx - frame_w * 0.5) / max(frame_w * 0.5, 1.0)
    lr = int(np.clip(-err_x * 78.0, -max_speed, max_speed))
    yaw = int(np.clip(err_x * 62.0, -max_speed, max_speed))
    fb = int(max_speed)
    return lr, fb, 0, yaw


def _draw_chase_hud(
    frame: np.ndarray,
    *,
    target: FaceTarget | None,
    chasing: bool,
    status: str,
    lr: int,
    fb: int,
    yaw: int,
    is_sim: bool,
) -> None:
    h, w = frame.shape[:2]
    if target is not None:
        x0, y0, x1, y1 = target.bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), _RED_BGR, 3)
        cv2.circle(frame, (target.cx, target.cy), 7, _RED_BGR, -1)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (60, 60, 90), 1)
    cmd = "CHASING" if chasing else status
    detail = f"RC {lr},{fb},{yaw}"
    draw_operator_hud(
        frame,
        mode="CHASE",
        command=cmd,
        detail=detail,
        is_sim=is_sim,
        mode_for_keys="chase",
        command_locked=chasing if chasing else None,
    )


def run_chase_loop(
    config: dict,
    *,
    webcam,
    insight,
    session_ctx=None,
) -> str | None:
    ccfg = config.get("chase", {}) or {}
    dcfg = config.get("drone", {}) or {}
    process_every = max(1, int(ccfg.get("process_every_n_frames", 2)))
    max_width = max(160, int(ccfg.get("max_width", 480)))
    max_speed = max(20, min(100, int(ccfg.get("max_rc_speed", 85))))
    hold_sec = float(ccfg.get("hold_seconds", 0.35))
    stop_size_ratio = float(ccfg.get("stop_size_ratio", 0.32))
    stale_sec = float(dcfg.get("fpv_stale_seconds", 5.0))

    owns_drone = session_ctx is None
    if session_ctx is not None:
        drone = session_ctx.ensure_drone(config)
    else:
        drone = DroneController(config)
        drone.connect()
    use_fpv = bool(dcfg.get("show_tello_fpv", False)) and drone.is_live_tello
    win = _CHASE_WINDOW if use_fpv else _SIM_WINDOW
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    hold = _SpaceHoldController(hold_seconds=hold_sec)
    hold.start()

    print(
        f"[chase] מקור: {'Tello FPV' if use_fpv else 'מצלמת מחשב (סימולציה)'}\n"
        "  פנים בפריים = מסגרת אדומה (הגדולה ביותר)\n"
        f"  החזק רווח (Space) ~{hold_sec:.1f}s+ = האצה מקסימלית | T המראה | L נחיתה"
    )
    print(f"[session] switch modes: {MODE_SWITCH_HINT}")

    frame_idx = 0
    last_target: FaceTarget | None = None
    last_good_video = time.monotonic()
    fpv_stale_warned = False
    requested_mode: str | None = None

    try:
        while True:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            raw = _read_frame(drone, webcam, use_fpv)
            if raw is not None:
                frame = raw
                last_good_video = time.monotonic()
                fpv_stale_warned = False
            else:
                frame = _blank_frame("No video frame")

            video_stale = (
                use_fpv
                and stale_sec > 0
                and drone.is_live_tello
                and (time.monotonic() - last_good_video) >= stale_sec
            )
            if video_stale and not fpv_stale_warned:
                print(f"[chase] FPV לא מתעדכן — עוצרים RC (>{stale_sec:.0f}s)")
                fpv_stale_warned = True

            chasing = hold.update()
            target: FaceTarget | None = None
            if not video_stale and frame is not None and frame.size > 0:
                if frame_idx % process_every == 0:
                    target = _detect_largest_face(insight, frame, max_width)
                    if target is not None:
                        last_target = target
                else:
                    target = last_target

            lr = fb = yaw = 0
            status = "SCAN"
            if target is None:
                status = "NO FACE"
                last_target = None
            elif chasing and drone.is_airborne and not video_stale:
                lr, fb, _, yaw = _compute_chase_rc(
                    target,
                    frame.shape[1],
                    max_speed=max_speed,
                    stop_size_ratio=stop_size_ratio,
                )
                if fb == 0 and target.size_ratio >= stop_size_ratio:
                    status = "TOO CLOSE — stop"
                else:
                    status = "RUSH"
            elif chasing and not drone.is_airborne:
                status = "GROUNDED — T to take off"
            elif target is not None:
                status = "TARGET LOCKED — hold SPACE"

            try:
                if chasing and drone.is_airborne and target is not None and not video_stale:
                    drone.send_rc(lr, fb, 0, yaw)
                else:
                    drone.send_rc(0, 0, 0, 0)
            except Exception as exc:
                print(f"[chase] RC: {exc}")

            _draw_chase_hud(
                frame,
                target=target,
                chasing=chasing and target is not None,
                status=status,
                lr=lr,
                fb=fb,
                yaw=yaw,
                is_sim=not drone.is_live_tello,
            )
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            requested = MODE_SWITCH_KEYS.get(key)
            if requested:
                requested_mode = requested
                break
            if key in (27, ord("q")):
                break
            if key in (ord("t"), ord("T")):
                try:
                    drone.takeoff()
                    print("[chase] takeoff")
                except Exception as exc:
                    print(f"[chase] takeoff: {exc}")
            if key in (ord("l"), ord("L")):
                try:
                    drone.send_rc(0, 0, 0, 0)
                    if drone.is_airborne:
                        drone.land()
                    print("[chase] land")
                except Exception as exc:
                    print(f"[chase] land: {exc}")

            frame_idx += 1
    finally:
        hold.stop()
        try:
            drone.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        if owns_drone:
            drone.disconnect()
        if session_ctx is not None:
            session_ctx.finish_mode_window(win, requested_mode)
        else:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
    return requested_mode


def _read_frame(drone: DroneController, webcam, use_fpv: bool) -> np.ndarray | None:
    if use_fpv:
        return drone.read_fpv_frame()
    if webcam is not None:
        return webcam.read_frame()
    return None


def _blank_frame(message: str) -> np.ndarray:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        img,
        message,
        (24, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return img
