"""On-screen face enrollment wizard for identity mode (no terminal)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from app.detection.face_landmarker_detector import FaceLandmarkerDetector
from app.identity.face_profile import (
    FaceProfileStore,
    average_embeddings,
    face_landmarks_to_embedding,
)
from app.ui.operator_hud import draw_operator_hud


def _draw_progress_bar(frame: np.ndarray, ratio: float, y_offset: int = 72) -> None:
    h, w = frame.shape[:2]
    bar_w = max(40, w - 80)
    bx, by, bh = 40, h - y_offset, 10
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bh), (50, 50, 65), -1)
    fill = int(bar_w * max(0.0, min(1.0, ratio)))
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (0, 200, 120), -1)


def _name_entry_loop(
    webcam,
    *,
    window_name: str,
    initial: str = "",
) -> str | None:
    buf = initial[:24]
    while True:
        frame = webcam.read_frame()
        if frame is None:
            return None
        cursor = "|" if int(time.monotonic() * 2) % 2 == 0 else " "
        draw_operator_hud(
            frame,
            mode="ENROLL",
            command=f"NAME: {buf}{cursor}",
            detail="who is flying today?",
            keys_line="type name  Backspace=erase  Enter=confirm",
            keys_line2="Esc=cancel",
        )
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(40) & 0xFF
        if k in (27, ord("q")):
            return None
        if k in (13, 10):
            name = buf.strip()
            return name if name else None
        if k in (8, 127):
            buf = buf[:-1]
        elif 32 <= k <= 126:
            ch = chr(k)
            if ch.isprintable() and len(buf) < 24:
                buf += ch


def _extract_embedding(
    frame: np.ndarray,
    *,
    embedding_backend: str,
    face_lm: FaceLandmarkerDetector | None,
    insight,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    eb = str(embedding_backend).lower()
    if eb == "insightface" and insight is not None:
        emb, bbox = insight.embed(frame)
        if bbox is not None:
            insight.draw_bbox(frame, bbox)
        return emb, bbox
    if face_lm is not None:
        res = face_lm.process_frame(frame)
        face_lm.draw_landmarks(frame, res)
        fl = face_lm.first_face_landmarks(res)
        if fl is None:
            return None, None
        emb = face_landmarks_to_embedding(list(fl))
        if emb is None:
            return None, None
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in fl]
        ys = [lm.y * h for lm in fl]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return emb, bbox
    return None, None


def _collect_loop(
    webcam,
    store: FaceProfileStore,
    *,
    display_name: str,
    enroll_frames: int,
    window_name: str,
    embedding_backend: str,
    face_lm: FaceLandmarkerDetector | None,
    insight,
) -> bool:
    collected: list[np.ndarray] = []
    eb = str(embedding_backend).lower()
    backend_id = "insightface" if eb == "insightface" else "landmarks"

    while len(collected) < enroll_frames:
        frame = webcam.read_frame()
        if frame is None:
            return False
        emb, bbox = _extract_embedding(
            frame,
            embedding_backend=embedding_backend,
            face_lm=face_lm,
            insight=insight,
        )
        good = emb is not None
        if good and emb is not None:
            collected.append(emb)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            color = (0, 220, 0) if good else (0, 80, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        ratio = len(collected) / max(enroll_frames, 1)
        _draw_progress_bar(frame, ratio)
        draw_operator_hud(
            frame,
            mode="ENROLL",
            command=f"CAPTURE {len(collected)}/{enroll_frames}",
            detail=display_name,
            keys_line="look at camera — hold still",
            keys_line2="Esc=cancel",
            command_locked=good,
        )
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            return False

    avg = average_embeddings(collected)
    if avg is None:
        return False
    store.save(avg, display_name, backend=backend_id)
    store.display_name = display_name
    return True


def _done_flash(webcam, *, window_name: str, name: str, seconds: float = 1.4) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = webcam.read_frame()
        if frame is None:
            break
        draw_operator_hud(
            frame,
            mode="ENROLL",
            command="SAVED",
            detail=name,
            keys_line="profile ready",
            command_locked=True,
        )
        cv2.imshow(window_name, frame)
        if (cv2.waitKey(40) & 0xFF) in (27, ord("q")):
            break


def run_identity_enroll_wizard(
    webcam,
    store: FaceProfileStore,
    *,
    embedding_backend: str,
    enroll_frames: int,
    window_name: str,
    face_lm: FaceLandmarkerDetector | None = None,
    insight=None,
    initial_name: str = "",
    reuse_window: bool = False,
) -> str | None:
    """
    Type name on screen, collect face frames, save profile.
    Returns saved name or None if cancelled.
    """
    if not reuse_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        default = initial_name or store.display_name or ""
        name = _name_entry_loop(webcam, window_name=window_name, initial=default)
        if not name:
            print("[identity] enroll cancelled (no name)")
            return None

        print(f"[identity] enrolling «{name}» — {enroll_frames} frames")
        ok = _collect_loop(
            webcam,
            store,
            display_name=name,
            enroll_frames=enroll_frames,
            window_name=window_name,
            embedding_backend=embedding_backend,
            face_lm=face_lm,
            insight=insight,
        )
        if not ok:
            print("[identity] enroll cancelled or failed")
            return None

        _done_flash(webcam, window_name=window_name, name=name)
        print(f"[identity] saved profile for «{name}»")
        return name
    except RuntimeError as exc:
        print(f"[identity] enroll error: {exc}")
        return None
    finally:
        if not reuse_window:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
