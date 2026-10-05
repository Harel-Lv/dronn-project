"""UDP client for dronn_rt_service (binary CommandPacket v1)."""
from __future__ import annotations

import socket
import struct
import time
from typing import Any

from app.drone.safety_rc import clamp_rc

_PROTOCOL_VERSION = 1
_CONNECT_ACK = bytes([0xAC])
_PACKET_FMT = "<BBH I Q hhhh"  # 24 bytes, little-endian
_PACKET_SIZE = struct.calcsize(_PACKET_FMT)

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
}


class RtCppClient:
    def __init__(self, cfg: dict[str, Any]) -> None:
        rt = cfg.get("rt_control", {})
        self._host = str(rt.get("host", "127.0.0.1"))
        self._port = int(rt.get("port", 9999))
        self._sock: socket.socket | None = None
        self._sequence = 0

    def connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1.5)
        self._send_named("connect")
        try:
            data, _addr = self._sock.recvfrom(8)
        except socket.timeout as exc:
            raise RuntimeError(
                "rt_cpp: no ACK from dronn_rt_service (is it running on the configured port?)"
            ) from exc
        if not data.startswith(_CONNECT_ACK):
            raise RuntimeError(f"rt_cpp: unexpected connect response: {data!r}")
        self._sock.settimeout(None)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._send_named("disconnect")
            except OSError:
                pass
            self._sock.close()
        self._sock = None

    def takeoff(self) -> None:
        self._send_named("takeoff")

    def land(self) -> None:
        self._send_named("land")

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        lr, fb, ud, yaw = clamp_rc(lr, fb, ud, yaw)
        self._send("rc_direct", lr=lr, fb=fb, ud=ud, yaw=yaw)

    def _send_named(self, name: str) -> None:
        self._send(name, lr=0, fb=0, ud=0, yaw=0)

    def _send(
        self,
        name: str,
        *,
        lr: int = 0,
        fb: int = 0,
        ud: int = 0,
        yaw: int = 0,
    ) -> None:
        if self._sock is None:
            raise RuntimeError("RtCppClient not connected")
        cmd = _CMD.get(name)
        if cmd is None:
            raise ValueError(f"unknown rt command: {name!r}")
        self._sequence += 1
        ts_us = int(time.time() * 1_000_000)
        packet = struct.pack(
            _PACKET_FMT,
            _PROTOCOL_VERSION,
            cmd,
            0,
            self._sequence,
            ts_us,
            int(lr),
            int(fb),
            int(ud),
            int(yaw),
        )
        if len(packet) != _PACKET_SIZE:
            raise RuntimeError(f"packet size mismatch: {len(packet)}")
        self._sock.sendto(packet, (self._host, self._port))
