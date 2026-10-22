"""Simulated or DJI Tello drone backend."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.drone.safety_rc import clamp_rc
from app.drone.tello_connect import connect_tello_and_read_battery
from app.drone.tello_network import format_tello_network_error, prepare_tello_network


@dataclass(frozen=True)
class FlightStatus:
    """Telemetry snapshot for the HUD (all None in simulation)."""

    live: bool
    link_lost: bool = False
    battery: int | None = None
    temp_c: int | None = None
    temp_hot: bool = False
    height_cm: int | None = None
    battery_low: bool = False


class DroneController:
    def __init__(self, config: dict) -> None:
        self._root_cfg = config
        self._cfg = config.get("drone", {})
        self._enabled = bool(self._cfg.get("enabled", False))
        self._backend = str(self._cfg.get("backend", "simulated")).lower()
        self._rt_client: Any = None
        self._tello: Any = None
        self._frame_reader: Any = None
        self._last_rc = (0, 0, 0, 0)
        self._connected = False
        self._last_battery: int | None = None
        self._last_battery_at: float = 0.0
        self._battery_display: float | None = None
        # Exponential moving average smoothing for HUD stability.
        self._battery_ema_alpha: float = 0.25
        self._airborne = False
        # Serializes SDK sends between the control loop and the keepalive thread;
        # held through blocking takeoff/land so keepalive RC never interleaves with them.
        self._cmd_lock = threading.RLock()
        self._last_cmd_at = time.monotonic()
        self._keepalive_stop: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._fpv_last_obj: Any = None
        self._fpv_last_bgr: np.ndarray | None = None
        self._fpv_last_new_at = 0.0
        self._fpv_started = False
        self._fpv_failed = False
        self._fpv_restarting = False
        self._fpv_restart_at = 0.0
        self._state_last_obj: Any = None
        self._state_last_at = 0.0
        self._link_lost = False
        self._battery_warned = False
        self._battery_autolanded = False

    @property
    def is_live_tello(self) -> bool:
        return self._enabled and self._backend == "tello"

    @property
    def uses_rt_cpp(self) -> bool:
        rt = self._root_cfg.get("rt_control", {})
        rt_on = bool(rt.get("enabled", False))
        return self._enabled and self._backend == "rt_cpp" and rt_on

    @property
    def is_airborne(self) -> bool:
        return self._airborne

    @property
    def has_fpv_stream(self) -> bool:
        return self._frame_reader is not None

    def sync_session_config(self, config: dict) -> None:
        """Apply per-mode config while keeping an open Tello session."""
        self._root_cfg = config
        self._cfg = config.get("drone", {})
        self._enabled = bool(self._cfg.get("enabled", False))
        self._backend = str(self._cfg.get("backend", "simulated")).lower()

    def ensure_fpv_stream(self) -> None:
        if not self.is_live_tello or self._tello is None:
            return
        if self._frame_reader is not None:
            return
        if not bool(self._cfg.get("show_tello_fpv", False)):
            return
        if self._fpv_failed and self._airborne:
            # A retry blocks for seconds; don't freeze every mode switch mid-flight.
            return
        old_retry = self._tello.retry_count
        old_timeout = self._tello.RESPONSE_TIMEOUT
        try:
            self._tello.retry_count = 1
            self._tello.RESPONSE_TIMEOUT = 4
            try:
                # Re-enter SDK: idle Tello drops command mode; streamon then
                # returns "Unkown Command!" and djitellopy retries ~28s.
                self._tello.send_control_command("command")
                time.sleep(0.25)
                self._tello.streamon()
            finally:
                self._tello.retry_count = old_retry
                self._tello.RESPONSE_TIMEOUT = old_timeout
            self._frame_reader = self._tello.get_frame_read()
            self._fpv_failed = False
            print("[drone] Tello FPV stream on")
        except Exception as exc:
            self._fpv_failed = True
            text = str(exc).lower()
            if "nknown" in text or "unkown" in text:
                print(
                    "[drone] FPV failed (Unknown command) — update firmware in the Tello "
                    "phone app. Flight control continues without drone video."
                )
            else:
                print(f"[drone] FPV stream failed: {exc}")

    def connect(self) -> None:
        if not self._enabled:
            print("[drone] SIMULATION — not connected to a real drone")
            self._connected = True
            return
        if self._backend == "rt_cpp":
            if not bool(self._root_cfg.get("rt_control", {}).get("enabled", False)):
                print(
                    "[drone] rt_cpp backend but rt_control.enabled=false — "
                    "using local simulation prints"
                )
                self._connected = True
                return
            from app.drone.rt_cpp_client import RtCppClient

            self._rt_client = RtCppClient(self._root_cfg)
            try:
                self._rt_client.connect()
            except (OSError, RuntimeError) as exc:
                self._rt_client = None
                raise RuntimeError(
                    "rt_cpp: dronn_rt_service unavailable — "
                    "start it first (see scripts/start_rt_service.py)"
                ) from exc
            print("[drone] rt_cpp — connected to real-time control service")
            self._connected = True
            self._start_keepalive()
            return
        if self._backend != "tello":
            print(f"[drone] backend={self._backend!r} — no network connection")
            self._connected = True
            return

        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError(
                "djitellopy is required for Tello: pip install djitellopy"
            ) from exc

        logging.getLogger("djitellopy").setLevel(logging.WARNING)

        net = prepare_tello_network()
        if not net.ok:
            raise RuntimeError(format_tello_network_error(net))

        retries = max(1, int(self._cfg.get("tello_connect_retries", 3)))
        delay = max(0.1, float(self._cfg.get("tello_connect_retry_delay_sec", 1.5)))
        # tello_host: null in YAML -> None; never pass host=None to djitellopy (breaks sendto).
        host_raw = self._cfg.get("tello_host")
        last_err: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                if host_raw is not None and str(host_raw).strip():
                    self._tello = Tello(host=str(host_raw).strip())
                else:
                    self._tello = Tello()
                self._tello.RESPONSE_TIMEOUT = max(
                    5, int(float(self._cfg.get("tello_response_timeout_sec", 8)))
                )
                self._battery_display = None
                b = connect_tello_and_read_battery(self._tello, self._cfg)
                self._last_battery = b
                self._last_battery_at = time.monotonic()
                print(f"[drone] Tello: battery={b}%")
                try:
                    self._tello.send_rc_control(0, 0, 0, 0)
                except Exception:
                    pass
                print(
                    "[drone] SDK link active. Fast yellow LED with app closed = no controller. "
                    "Slow yellow on a table = Attitude mode (normal)."
                )
                self.ensure_fpv_stream()
                self._connected = True
                self._start_keepalive()
                return
            except Exception as exc:
                last_err = exc
                if self._tello is not None:
                    try:
                        self._tello.end()
                    except Exception:
                        pass
                    self._tello.stream_on = False
                    self._tello.is_flying = False
                self._tello = None
                self._frame_reader = None
                if attempt < retries:
                    print(
                        f"[drone] connect failed ({attempt}/{retries}): {exc} — "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"Tello: connect failed after {retries} attempts. Last error: {last_err}"
        ) from last_err

    def safe_land_if_airborne(self, *, reason: str = "") -> None:
        """Zero RC and land when the session ends while the drone is still airborne."""
        if not self._airborne:
            return
        suffix = f" ({reason})" if reason else ""
        if self.is_live_tello or self._rt_client is not None:
            print(f"[drone] SAFETY LANDING{suffix}...")
        try:
            self.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        try:
            self.land()
        except Exception as exc:
            print(f"[drone] SAFETY LANDING FAILED: {exc}")
        else:
            self._airborne = False

    def disconnect(self, *, land: bool | None = None) -> None:
        do_land = (
            bool(self._cfg.get("land_on_disconnect", True))
            if land is None
            else bool(land)
        )
        if do_land:
            self.safe_land_if_airborne(reason="session ended")
        self._stop_keepalive()
        if self._rt_client is not None:
            try:
                self.send_rc(0, 0, 0, 0)
            except Exception:
                pass
            try:
                self._rt_client.close()
            except Exception:
                pass
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
            # Tello.__del__ calls end() again; stale flags make it hit a removed host (KeyError).
            self._tello.stream_on = False
            self._tello.is_flying = False
        self._rt_client = None
        self._tello = None
        self._frame_reader = None
        self._connected = False
        self._battery_display = None

    def takeoff(self) -> None:
        if self._airborne:
            if self._tello is not None and self._reported_height_cm() == 0:
                # Tello auto-lands after 15s without commands; resync instead of blocking takeoff.
                print("[drone] drone reports height 0 (it landed on its own) — taking off")
                self._airborne = False
            else:
                print("[drone] already airborne — takeoff ignored")
                return
        if self._rt_client is not None:
            with self._cmd_lock:
                self._rt_client.takeoff()
                self._last_cmd_at = time.monotonic()
            self._airborne = True
            return
        if self._tello is None:
            print("[drone:sim] TAKEOFF")
            self._airborne = True
            return
        with self._cmd_lock:
            # djitellopy retries takeoff 3x with a 20s timeout each, freezing the control loop.
            old_retry = self._tello.retry_count
            self._tello.retry_count = 1
            try:
                self._tello.takeoff()
            except Exception as exc:
                # Reply lost but the drone may be flying: without the flag, L and auto-land
                # on exit would both skip it.
                h = self._reported_height_cm()
                if h is not None and h > 0:
                    print(f"[drone] takeoff reply lost but drone reports height {h}cm — marked airborne")
                    self._airborne = True
                    return
                reason = self._takeoff_refusal_reason()
                if reason:
                    raise RuntimeError(f"Tello refused takeoff: {reason}") from exc
                raise
            finally:
                self._tello.retry_count = old_retry
                self._last_cmd_at = time.monotonic()
        self._airborne = True

    def land(self) -> None:
        unsure = False
        if not self._airborne:
            # Flag can be out of sync with a real Tello; only skip when it reports ground.
            if self._tello is None or self._reported_height_cm() == 0:
                return
            print("[drone] not marked airborne but drone may be flying — sending land anyway")
            unsure = True
        if self._rt_client is not None:
            with self._cmd_lock:
                self._rt_client.land()
                self._last_cmd_at = time.monotonic()
            self._airborne = False
            return
        if self._tello is None:
            print("[drone:sim] LAND")
            self._airborne = False
            return
        with self._cmd_lock:
            try:
                self._tello.land()
            except Exception:
                # Tello answers 'error' to land when it is already on the ground.
                if unsure or self._reported_height_cm() == 0:
                    print("[drone] land refused — drone is on the ground, marked as landed")
                    self._airborne = False
                    return
                raise
            finally:
                self._last_cmd_at = time.monotonic()
        self._airborne = False

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        lr, fb, ud, yaw = clamp_rc(lr, fb, ud, yaw)
        self._last_rc = (lr, fb, ud, yaw)
        if self._rt_client is None and self._tello is None:
            if any(self._last_rc):
                print(f"[drone:sim] RC {self._last_rc}")
            return
        with self._cmd_lock:
            self._send_rc_raw(lr, fb, ud, yaw)
            self._last_cmd_at = time.monotonic()

    def _send_rc_raw(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self._rt_client is not None:
            self._rt_client.send_rc(lr, fb, ud, yaw)
        elif self._tello is not None:
            self._tello.send_rc_control(lr, fb, ud, yaw)

    def _reported_height_cm(self) -> int | None:
        """Height from Tello state stream, else 'height?' query. None if unknown."""
        if self._tello is None:
            return None
        try:
            state = self._tello.get_current_state()
            if state and "h" in state:
                return int(state["h"])
        except Exception:
            pass
        with self._cmd_lock:
            try:
                resp = str(self._tello.send_command_with_return("height?", timeout=3))
            except Exception:
                return None
        low = resp.lower()
        if "abort" in low or "error" in low:
            return None
        digits = "".join(ch for ch in resp if ch.isdigit())
        if not digits:
            return None
        value = int(digits)
        return value * 10 if "dm" in low else value

    def _start_keepalive(self) -> None:
        """Tello auto-lands after 15s without commands (mode loading, calibration, identity mode)."""
        if self._keepalive_thread is not None:
            return
        idle_sec = max(1.0, float(self._cfg.get("keepalive_idle_sec", 1.5)))
        stop = threading.Event()

        def run() -> None:
            while not stop.wait(0.5):
                if not self._airborne:
                    continue
                if time.monotonic() - self._last_cmd_at < idle_sec:
                    continue
                if not self._cmd_lock.acquire(blocking=False):
                    continue
                try:
                    self._send_rc_raw(0, 0, 0, 0)
                    self._last_cmd_at = time.monotonic()
                except Exception:
                    pass
                finally:
                    self._cmd_lock.release()

        self._keepalive_stop = stop
        self._keepalive_thread = threading.Thread(target=run, name="drone-keepalive", daemon=True)
        self._keepalive_thread.start()

    def _stop_keepalive(self) -> None:
        if self._keepalive_stop is not None:
            self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=1.5)
        self._keepalive_stop = None
        self._keepalive_thread = None

    def _state_int(self, key: str) -> int | None:
        if self._tello is None:
            return None
        try:
            state = self._tello.get_current_state()
            return int(float(state[key])) if state and key in state else None
        except Exception:
            return None

    def _takeoff_refusal_reason(self) -> str | None:
        """Tello only answers 'error'; explain the usual causes from telemetry."""
        block_pct = int(self._cfg.get("battery_block_percent", 15))
        hot_c = int(self._cfg.get("temp_warn_c", 85))
        bat = self._state_int("bat")
        temp = self._state_int("temph")
        if bat is not None and bat < block_pct:
            return f"battery {bat}% is too low — charge it"
        if temp is not None and temp >= hot_c:
            return (
                f"drone is too hot ({temp}C) — let it cool a few minutes "
                "(it heats up when powered on without flying)"
            )
        return None

    def poll_flight_status(self) -> FlightStatus:
        """
        Call once per frame from the control loop: detects telemetry loss and lands on
        critical battery (the land runs on the caller's thread, like pressing L).
        """
        if not self.is_live_tello or self._tello is None:
            return FlightStatus(live=False)
        now = time.monotonic()
        try:
            state = self._tello.get_current_state()
        except Exception:
            state = None
        # djitellopy replaces the state dict on every packet (~10 Hz).
        if state and state is not self._state_last_obj:
            self._state_last_obj = state
            self._state_last_at = now
        lost_sec = max(0.5, float(self._cfg.get("link_lost_seconds", 2.0)))
        # No state ever (UDP 8890 blocked): can't tell, so don't claim the link is lost.
        lost = self._state_last_at > 0 and now - self._state_last_at > lost_sec
        if lost and not self._link_lost:
            print(
                f"[drone] LINK LOST — no telemetry for {lost_sec:g}s+. "
                "Tello hovers and auto-lands if it hears nothing; stay clear."
            )
        elif not lost and self._link_lost:
            print("[drone] link restored")
        self._link_lost = lost

        battery = self.get_battery_percent(min_interval_sec=1.0)
        temp = self._state_int("temph")
        height = self._state_int("h")
        hot_c = int(self._cfg.get("temp_warn_c", 85))
        warn_pct = int(self._cfg.get("battery_inflight_warn_percent", 20))
        land_pct = int(self._cfg.get("battery_autoland_percent", 10))
        low = battery is not None and battery <= warn_pct

        if not self._airborne:
            # Re-arm so taking off again on a critical battery lands again.
            self._battery_autolanded = False
        elif battery is not None and not lost:
            if battery <= land_pct and not self._battery_autolanded:
                self._battery_autolanded = True
                print(f"[drone] BATTERY {battery}% — AUTO-LANDING")
                try:
                    self.send_rc(0, 0, 0, 0)
                    self.land()
                except Exception as exc:
                    print(f"[drone] battery auto-land failed: {exc} — press L")
            elif low and not self._battery_warned:
                self._battery_warned = True
                print(f"[drone] WARNING: battery {battery}% — land soon (auto-land at {land_pct}%)")

        return FlightStatus(
            live=True,
            link_lost=lost,
            battery=battery,
            temp_c=temp,
            temp_hot=temp is not None and temp >= hot_c,
            height_cm=height,
            battery_low=low,
        )

    def read_fpv_frame(self) -> np.ndarray | None:
        """
        Latest drone frame as BGR, or None if there is no live video.
        djitellopy never returns None: it serves a black placeholder before decoding starts
        and keeps the last frame forever when the stream freezes, so both are detected here.
        """
        if self._frame_reader is None:
            return None
        now = time.monotonic()
        self._fpv_watchdog(now)
        try:
            f = self._frame_reader.frame
        except Exception:
            return None
        if f is None or f.size == 0:
            return None
        if f is not self._fpv_last_obj:
            self._fpv_last_obj = f
            self._fpv_last_new_at = now
            if not self._fpv_started:
                is_placeholder = f.shape[:2] == (300, 400) and not f.any()
                if is_placeholder:
                    return None
                self._fpv_started = True
            # djitellopy decodes via PIL -> RGB; the rest of the app (OpenCV, MediaPipe wrappers,
            # InsightFace) expects BGR.
            self._fpv_last_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        if not self._fpv_started:
            return None
        freeze_sec = max(0.3, float(self._cfg.get("fpv_freeze_seconds", 1.0)))
        if now - self._fpv_last_new_at > freeze_sec:
            return None
        # Callers draw HUD on the frame; the loop runs faster than video, so the same
        # frame is returned repeatedly and must not carry previous overlays into detection.
        return self._fpv_last_bgr.copy()

    def _fpv_watchdog(self, now: float) -> None:
        """
        djitellopy's decoder thread dies for good on a UDP read error (av.error.OSError) and
        the reader keeps serving the last frame, so reopen the decoder in the background.
        """
        reader = self._frame_reader
        worker = getattr(reader, "worker", None)
        if worker is None or worker.is_alive() or self._fpv_restarting:
            return
        if now - self._fpv_restart_at < 3.0:
            return
        self._fpv_restart_at = now
        self._fpv_restarting = True
        threading.Thread(target=self._restart_fpv_decoder, args=(reader,), daemon=True).start()

    def _restart_fpv_decoder(self, dead_reader: Any) -> None:
        try:
            tello = self._tello
            if tello is None or self._frame_reader is not dead_reader:
                return
            print("[drone] FPV decoder stopped (video read error) - reopening stream...")
            try:
                dead_reader.stop()
                dead_reader.container.close()
            except Exception:
                pass
            # Only the decoder is reopened: the Tello keeps streaming after streamon, and SDK
            # commands from this thread would block / mix replies with the control loop.
            tello.background_frame_read = None
            reader = tello.get_frame_read()
            self._fpv_started = False
            self._fpv_last_obj = None
            self._frame_reader = reader
            print("[drone] FPV stream reopened")
        except Exception as exc:
            print(f"[drone] FPV reopen failed: {exc} - retrying")
        finally:
            self._fpv_restarting = False

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
