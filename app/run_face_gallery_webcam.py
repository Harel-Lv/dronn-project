"""Test face gallery matching on the PC webcam (no drone)."""
from __future__ import annotations

import cv2

from app.detection.face_landmarker_detector import FaceLandmarkerDetector
from app.identity.face_gallery import face_gallery_from_config
from app.identity.face_profile import face_landmarks_to_embedding


def run_face_gallery_webcam_loop(webcam, config: dict) -> None:
    """Multi-person gallery on PC webcam (not the same as --mode identity)."""
    win = "Face gallery (webcam)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    gallery = face_gallery_from_config(config)
    fcfg = config.get("fpv_faces", {})
    backend = str(fcfg.get("embedding_backend", "landmarks")).lower()
    n_faces = len(gallery)

    face_if = None
    face_lm = None
    if backend == "insightface":
        try:
            from app.identity.insightface_backend import InsightFaceBackend

            face_if = InsightFaceBackend(config)
            print("[gallery] InsightFace backend")
        except Exception as exc:
            print(f"[gallery] InsightFace failed ({exc}), falling back to landmarks")
            face_lm = FaceLandmarkerDetector()
    else:
        face_lm = FaceLandmarkerDetector()

    print(f"[gallery] profiles: {n_faces} | q/Esc quit")

    try:
        while True:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame = webcam.read_frame()
            if frame is None:
                break

            label = "—"
            score = 0.0
            if face_if is not None:
                emb, bbox = face_if.embed(frame)
                face_if.draw_bbox(frame, bbox)
                if emb is not None:
                    name, score = gallery.best_match(emb)
                    label = name or "לא מזוהה"
            elif face_lm is not None:
                res = face_lm.process_frame(frame)
                face_lm.draw_landmarks(frame, res)
                fl = face_lm.first_face_landmarks(res)
                emb = face_landmarks_to_embedding(list(fl)) if fl is not None else None
                if emb is not None:
                    name, score = gallery.best_match(emb)
                    label = name or "לא מזוהה"

            cv2.putText(
                frame,
                f"{label} ({score:.2f})",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if label != "לא מזוהה" and label != "—" else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        if face_lm is not None:
            face_lm.close()
        if face_if is not None:
            face_if.close()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
