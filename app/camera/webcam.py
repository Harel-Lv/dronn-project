import sys

import cv2


class Webcam:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def open(self) -> None:
        # Use DirectShow on Windows for better compatibility; use default on other platforms.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Webcam is not open")

        success, frame = self.cap.read()
        if not success:
            return None

        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None