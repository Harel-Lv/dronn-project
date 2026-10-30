"""UDP client for dronn_rt_service (binary CommandPacket v1 + CommandAckPacket)."""
from __future__ import annotations

import socket
import struct
import time
from typing import Any

from app.drone.safety_rc import clamp_rc

_PROTOCOL_VERSION = 1
_ACK_MARKER = 0xAC
_PACKET_FMT = "<BBH I Q hhhh"  # 24 bytes, little-endian
_ACK_FMT = "<BBBB I"  # marker, version, flags, flight_state, sequence
_PACKET_SIZE = struct.calcsize(_PACKET_FMT)
_ACK_SIZE = struct.calcsize(_ACK_FMT)

_CMD = {
    "takeoff": 1,
    "land": 2,
    "hover": 3,
    "forward": 4,
    "back": 5,
    "left": 6,
    "right": 7,
    "up": 8,
    "down": 9,
    "rotate_cw": 10,
    "rotate_ccw": 11,
    "rc_direct": 12,
    "connect": 13,
    "disconnect": 14,
    "recover": 15,
}

_CRITICAL = frozenset({"connect", "takeoff", "land", "disconnect", "recover"})


class RtCppAckError(RuntimeError):
    """Backend or FSM rejected a critical command."""

    def __init__(self, command: str, flight_state: int | None = None) -> None:
        self.command = command
        self.flight_state = flight_state
        msg = f"rt_cpp: {command} rejected"
        if flight_state is not None:
            msg += f" (flight_state={flight_state})"
        super().__init__(msg)


class RtCppClient:
    def __init__(self, cfg: dict[str, Any]) -> None:
        rt = cfg.get("rt_control", {})
        self._host = str(rt.get("host", "127.0.0.1"))
        self._port = int(rt.get("port", 9999))
        self._ack_timeout_s = float(rt.get("ack_timeout_s", 2.5))
        self._sock: socket.socket | None = None
        self._sequence = 0
        self._last_flight_state: int | None = None

    @property
    def last_flight_state(self) -> int | None:
        return self._last_flight_state

    def connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self._ack_timeout_s)
        self._send_named("connect", wait_ack=True)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._send_named("disconnect", wait_ack=True)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    def takeoff(self) -> None:
        self._send_named("takeoff", wait_ack=True)

    def land(self) -> None:
        self._send_named("land", wait_ack=True)

    def recover(self) -> None:
        """Clear safety latch after watchdog failure (requires grounded backend)."""
        self._send_named("recover", wait_ack=True)

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        lr, fb, ud, yaw = clamp_rc(lr, fb, ud, yaw)
        self._send("rc_direct", lr=lr, fb=fb, ud=ud, yaw=yaw, wait_ack=False)

    def _send_named(self, name: str, *, wait_ack: bool = False) -> None:
        self._send(name, lr=0, fb=0, ud=0, yaw=0, wait_ack=wait_ack)

    def _parse_ack(self, data: bytes) -> tuple[int, int, int, int, int] | None:
        if len(data) != _ACK_SIZE:
            return None
        marker, ver, flags, fstate, seq = struct.unpack(_ACK_FMT, data)
        if marker != _ACK_MARKER or ver != _PROTOCOL_VERSION:
            return None
        return marker, ver, flags, fstate, seq

    def _wait_ack(self, expected_seq: int, command: str) -> None:
        if self._sock is None:
            raise RuntimeError("RtCppClient not connected")
        deadline = time.monotonic() + self._ack_timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._sock.settimeout(remaining)
            try:
                data, _addr = self._sock.recvfrom(64)
            except socket.timeout:
                break
            except OSError as exc:
                raise RuntimeError(f"rt_cpp: ACK recv failed for {command}") from exc
            parsed = self._parse_ack(data)
            if parsed is None:
                continue
            _m, _v, flags, fstate, seq = parsed
            if seq != expected_seq:
                continue
            self._last_flight_state = fstate
            if flags != 0:
                raise RtCppAckError(command, fstate)
            return
        raise RuntimeError(
            f"rt_cpp: no ACK for {command} seq={expected_seq} "
            f"(is dronn_rt_service running on {self._host}:{self._port}?)"
        )

    def _send(
        self,
        name: str,
        *,
        lr: int = 0,
        fb: int = 0,
        ud: int = 0,
        yaw: int = 0,
        wait_ack: bool = False,
    ) -> None:
        if self._sock is None:
            raise RuntimeError("RtCppClient not connected")
        cmd = _CMD.get(name)
        if cmd is None:
            raise ValueError(f"unknown rt command: {name!r}")
        self._sequence += 1
        seq = self._sequence
        ts_us = int(time.time() * 1_000_000)
        packet = struct.pack(
            _PACKET_FMT,
            _PROTOCOL_VERSION,
            cmd,
            0,
            seq,
            ts_us,
            int(lr),
            int(fb),
            int(ud),
            int(yaw),
        )
        if len(packet) != _PACKET_SIZE:
            raise RuntimeError(f"packet size mismatch: {len(packet)}")
        self._sock.sendto(packet, (self._host, self._port))
        if name in _CRITICAL or wait_ack:
            self._wait_ack(seq, name)
