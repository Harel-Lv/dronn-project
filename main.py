"""CLI entry point — use launcher.py for the graphical UI."""
import argparse

from app.session import MODE_CHOICES, SessionOptions, run_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone Gesture Control / Target Tracking")
    parser.add_argument(
        "--mode",
        choices=list(MODE_CHOICES),
        default="manual",
        help=(
            "gesture=hands only | identity=single face profile | "
            "tracking=body follow (FPV+pose) | chase=face rush on Space hold | "
            "manual=keyboard (Tello) | webcam_faces=identity gate + gestures"
        ),
    )
    parser.add_argument(
        "--tello",
        action="store_true",
        help="Connect to real DJI Tello (overrides config: drone.enabled=true, backend=tello)",
    )
    parser.add_argument(
        "--fpv",
        action="store_true",
        help="Tello live camera in a second window. Works with --tello + gesture/webcam_faces/tracking",
    )
    parser.add_argument(
        "--fpv-faces",
        action="store_true",
        help="On --mode manual: match FPV faces to gallery",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default 0)",
    )
    args = parser.parse_args()
    run_session(SessionOptions.from_namespace(args))


if __name__ == "__main__":
    main()
