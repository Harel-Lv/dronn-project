"""Resilient Tello connect (command port + state or battery? fallback)."""
from __future__ import annotations

import time
from typing import Any


def connect_tello_and_read_battery(tello: Any, drone_cfg: dict[str, Any]) -> int:
    """
    Enter SDK mode without failing immediately on missing state UDP.
    Prefer state stream; fall back to battery? query (works when 8890 is blocked).
    """
    wait_sec = max(2.0, float(drone_cfg.get("tello_state_wait_sec", 8.0)))
    tello.connect(wait_for_state=False)
    deadline = time.monotonic() + wait_sec
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        state = tello.get_current_state()
        if state and "bat" in state:
            return int(state["bat"])
        try:
            return int(tello.query_battery())
        except Exception as exc:
            last_err = exc
        time.sleep(0.2)
    msg = (
        "Did not receive Tello state (UDP 8890) or battery? response. "
        "Check Wi‑Fi Tello-XXXXXX and allow Python through Windows Firewall (Private)."
    )
    if last_err is not None:
        raise RuntimeError(f"{msg} Last error: {last_err}") from last_err
    raise RuntimeError(msg)
