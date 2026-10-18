"""Detect / fix Windows routing to Tello (192.168.10.1)."""
from __future__ import annotations

import re
import socket
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TelloNetworkStatus:
    route_source_ip: str | None
    tello_subnet_ips: tuple[str, ...]
    on_tello_subnet: bool

    @property
    def ok(self) -> bool:
        return self.on_tello_subnet


def _collect_tello_subnet_ips() -> list[str]:
    ips: list[str] = []
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-NetIPAddress -AddressFamily IPv4 | "
                    "Where-Object { $_.IPAddress -like '192.168.10.*' } | "
                    "Select-Object -ExpandProperty IPAddress",
                ],
                text=True,
                errors="replace",
                timeout=8,
            )
            for line in out.splitlines():
                ip = line.strip()
                if ip.startswith("192.168.10."):
                    ips.append(ip)
        except (OSError, subprocess.SubprocessError):
            pass
    if not ips:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if ip.startswith("192.168.10."):
                    ips.append(ip)
        except OSError:
            pass
    if not ips:
        try:
            out = subprocess.check_output(["ipconfig"], text=True, errors="replace", timeout=8)
            ips.extend(re.findall(r"\b(192\.168\.10\.\d{1,3})\b", out))
        except (OSError, subprocess.SubprocessError):
            pass
    return sorted(set(ips))


def _route_source_to(host: str = "192.168.10.1", port: int = 8889) -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((host, port))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def check_tello_network() -> TelloNetworkStatus:
    tello_ips = tuple(_collect_tello_subnet_ips())
    route_ip = _route_source_to()
    on_subnet = bool(route_ip and route_ip.startswith("192.168.10."))
    return TelloNetworkStatus(
        route_source_ip=route_ip,
        tello_subnet_ips=tello_ips,
        on_tello_subnet=on_subnet,
    )


def _try_windows_host_route(local_tello_ip: str) -> bool:
    if sys.platform != "win32":
        return False
    try:
        subprocess.run(
            [
                "route",
                "ADD",
                "192.168.10.0",
                "MASK",
                "255.255.255.0",
                local_tello_ip,
                "METRIC",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def prepare_tello_network(*, try_fix_route: bool = True) -> TelloNetworkStatus:
    """
    Ensure UDP to 192.168.10.1 uses the Tello Wi-Fi subnet.
    On Windows with multiple adapters, may add a host route (no admin if route is new).
    """
    status = check_tello_network()
    if status.ok:
        return status
    if try_fix_route and status.tello_subnet_ips:
        local_ip = status.tello_subnet_ips[0]
        if _try_windows_host_route(local_ip):
            status = check_tello_network()
            if status.ok:
                print(f"[drone] Tello route fixed via {local_ip}")
                return status
    return status


def format_tello_network_error(status: TelloNetworkStatus) -> str:
    lines = [
        "Tello: no SDK path to 192.168.10.1 (UDP 8889).",
    ]
    if status.route_source_ip:
        lines.append(f"  PC route source IP: {status.route_source_ip} (need 192.168.10.x)")
    else:
        lines.append("  No route to 192.168.10.1")
    if status.tello_subnet_ips:
        lines.append(
            f"  Tello Wi-Fi NIC has {', '.join(status.tello_subnet_ips)} "
            "but Windows sends traffic elsewhere — disconnect Ethernet/VPN, "
            "or run verify_tello as Administrator once."
        )
    else:
        lines.append(
            "  Wi-Fi must be TELLO-XXXXXX only; ipconfig should show 192.168.10.x on Wi-Fi."
        )
    lines.append("  Then: python scripts/verify_tello.py")
    return "\n".join(lines)
