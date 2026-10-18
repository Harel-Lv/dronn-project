#!/usr/bin/env python3
"""Quick Tello connectivity check (no takeoff)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import load_config
from app.drone.tello_network import format_tello_network_error, prepare_tello_network
from app.ui.preflight import invalidate_tello_probe_cache, probe_tello


def _local_wifi_hint() -> tuple[str, bool]:
    net = prepare_tello_network()
    lines = []
    if net.route_source_ip:
        lines.append(f"Route source IP to 192.168.10.1: {net.route_source_ip}")
    if net.tello_subnet_ips:
        lines.append(f"Tello subnet NIC(s): {', '.join(net.tello_subnet_ips)}")
    if not net.ok:
        lines.append(format_tello_network_error(net))
    return "\n".join(lines), net.ok


def main() -> None:
    invalidate_tello_probe_cache()
    cfg = load_config()
    cfg.setdefault("drone", {})
    cfg["drone"]["enabled"] = True
    cfg["drone"]["backend"] = "tello"
    hint, on_subnet = _local_wifi_hint()
    print(hint)
    print("Probing Tello (djitellopy, may take ~10s) ...")
    result = probe_tello(config=cfg, live=True)
    try:
        print(result.detail)
    except UnicodeEncodeError:
        print(result.detail.encode("ascii", errors="replace").decode("ascii"))
    if not result.connected or not on_subnet:
        print(
            "Tips: Wi-Fi TELLO-XXXXXX ONLY (ipconfig -> 192.168.10.x) | "
            "disconnect Ethernet/VPN (e.g. vnet-devtest) | drone powered | "
            "Windows Firewall: allow Python on Private | close Tello phone app"
        )
        sys.exit(1)
    print("OK - run: python main.py --mode gesture --tello")


if __name__ == "__main__":
    main()
