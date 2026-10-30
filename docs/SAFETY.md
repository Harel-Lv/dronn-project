# Safety model (dronn-project)

This is a **hobby / learning project**, not a certified or commercial drone product.  
Operators are responsible for safe flight, local laws, and keeping people clear of the aircraft.

## Two control paths

| Path | When used | Notes |
|------|-----------|--------|
| **Python + Tello** (`drone.backend: tello`) | Real DJI Tello flights | Primary production path. Uses `djitellopy` (UDP SDK), not the 24-byte `rt_control` packet. |
| **Python + rt_cpp** | Demo / architecture study | UDP to `dronn_rt_service` → **simulator backend only** today. Not validated on physical hardware. |

Computer vision (gestures, tracking, chase) runs in Python. Safety logic differs slightly per path but shares ideas: clamp RC, hover on timeout, block motion when inappropriate.

## Python (`DroneController`)

- **Keepalive:** periodic `send_rc(0,0,0,0)` while airborne so Tello does not auto-land after ~15 s silence.
- **Emergency land:** **L** in gesture/manual/tracking/chase modes.
- **Quit:** **Esc** always exits; **q** blocked while airborne.
- **Telemetry HUD:** link lost, battery warn, low-battery auto-land (config thresholds), motor temperature.
- **FPV:** stale/frozen stream treated as no frame; manual keyboard still flies without video; FPV decoder can reopen after `djitellopy` read errors.
- **Takeoff / land:** resync with Tello height when SDK replies are lost; refusal when battery/temp blocks takeoff.
- **Gestures:** intent stabilizer (extra frames for TAKEOFF/LAND); enrolled-face gate in `webcam_faces`.
- **Manual keys:** `pynput` is global; RC only when the session OpenCV window is focused.

## C++ (`rt_control`)

Real-time-**oriented** periodic loop (default 20 Hz) on Linux/WSL — **not** hard real-time.

- **Flight state machine:** motion only in `FLYING` / `HOVERING`; takeoff only from `READY` after backend connect; `READY` only after successful `backend->connect()`.
- **Command queue:** critical commands (`CONNECT`, `DISCONNECT`, `TAKEOFF`, `LAND`) kept in a separate slot so high-rate RC cannot overwrite them.
- **Watchdog:**
  - **Hover** — no valid command for `watchdog_hover_ms` → zero RC (simulated hover).
  - **Safe** — no command for `watchdog_safe_ms` → `emergency_stop()` then **controlled** `land()` if backend reports success; FSM reflects land failure if not.
- **UDP ingress:** version/size checks, RC clamp, stale/future timestamps (see `rt_control/README.md`).

Simulator success does **not** prove Tello behavior.

## What is not guaranteed

- No guarantee against Wi‑Fi dropouts, SDK bugs, or mis-detected gestures.
- No geofencing, obstacle avoidance, or redundant hardware failsafes.
- InsightFace / MediaPipe can misclassify; always supervise the drone.

## Recommended preflight

1. `python scripts/smoke_test.py`
2. For Tello: `python scripts/verify_tello.py` on `Tello-XXXXXX` Wi‑Fi only.
3. Open area, battery above config block threshold, motors cool.
4. Test **L** on the ground, then short hover before gesture/tracking.

## Reporting issues

Document: mode, config (`drone` section), terminal log, whether FPV was active, and if link/battery HUD showed warnings.
