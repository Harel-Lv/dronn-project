"""Preview loop: single enrolled face — landmarks or InsightFace ArcFace."""
from __future__ import annotations

import cv2
import numpy as np

from app.detection.face_landmarker_detector import FaceLandmarkerDetector
from app.identity.face_profile import FaceProfileStore, face_landmarks_to_embedding
from app.run_gesture_session import run_face_enrollment, run_face_enrollment_insightface
from app.ui.unicode_text import draw_text_utf8


def _landmarks_to_bbox(
    frame: np.ndarray,
    landmarks_list: list,
    *,
    pad_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks_list]
    ys = [lm.y * h for lm in landmarks_list]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = x1 - x0
    dy = y1 - y0
    pad_x = max(dx * pad_ratio, 6.0)
    pad_y = max(dy * pad_ratio, 6.0)
    xi0 = int(max(0, x0 - pad_x))
    yi0 = int(max(0, y0 - pad_y))
    xi1 = int(min(w - 1, x1 + pad_x))
    yi1 = int(min(h - 1, y1 + pad_y))
    return xi0, yi0, xi1, yi1


def _draw_no_face_overlay(frame: np.ndarray, window_name: str) -> int:
    """Show 'no face', return waitKey code."""
    draw_text_utf8(
        frame,
        "אין פנים במסגרת",
        (12, 8),
        font_px=22,
        color_bgr=(160, 160, 160),
    )
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF


def run_identity_preview_loop(
    webcam,
    store: FaceProfileStore,
    *,
    embedding_backend: str,
    match_threshold: float,
    enroll_frames: int = 25,
    window_name: str = "Identity",
    enroll_window_name: str = "Identity — enroll",
    face_lm: FaceLandmarkerDetector | None = None,
    insight=None,
) -> None:
    eb = str(embedding_backend).lower()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    backend_label = "ArcFace (InsightFace)" if eb == "insightface" else "MediaPipe landmarks"
    print(
        f"[identity] תצוגה — {backend_label} | סף {match_threshold:.3f}\n"
        f"  q / Esc — יציאה | N — רישום מחדש (שם בטרמינל, {enroll_frames} פריימים)"
    )

    def _handle_reenroll_key() -> None:
        _try_reenroll_from_terminal(
            webcam,
            store,
            embedding_backend=eb,
            enroll_frames=enroll_frames,
            enroll_window_name=enroll_window_name,
            face_lm=face_lm,
            insight=insight,
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame = webcam.read_frame()
            if frame is None:
                break

            ok = False
            score = 0.0
            stored_name = store.display_name
            x0 = y0 = x1 = y1 = 0

            if eb == "insightface" and insight is not None:
                emb, bbox = insight.embed(frame)
                if bbox is None or emb is None:
                    k = _draw_no_face_overlay(frame, window_name)
                    if k in (27, ord("q")):
                        break
                    if k in (ord("n"), ord("N")):
                        _handle_reenroll_key()
                    continue
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                ok, score, stored_name = store.verify(emb, float(match_threshold))
            elif face_lm is not None:
                res = face_lm.process_frame(frame)
                fl = face_lm.first_face_landmarks(res)
                if fl is None:
                    k = _draw_no_face_overlay(frame, window_name)
                    if k in (27, ord("q")):
                        break
                    if k in (ord("n"), ord("N")):
                        _handle_reenroll_key()
                    continue
                emb = face_landmarks_to_embedding(list(fl))
                ok, score, stored_name = store.verify(emb, float(match_threshold))
                x0, y0, x1, y1 = _landmarks_to_bbox(frame, list(fl))
            else:
                draw_text_utf8(
                    frame,
                    "שגיאה: אין מודל פנים",
                    (12, 8),
                    font_px=22,
                    color_bgr=(0, 0, 255),
                )
                cv2.imshow(window_name, frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                continue

            color = (0, 220, 0) if ok else (0, 80, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            if ok:
                label = f"{stored_name}  ({score:.2f})"
                draw_text_utf8(
                    frame,
                    label,
                    (x0, max(4, y0 - 36)),
                    font_px=26,
                    color_bgr=color,
                )
            else:
                unknown = f"לא ידוע ({score:.2f})"
                draw_text_utf8(
                    frame,
                    unknown,
                    (x0, max(4, y0 - 36)),
                    font_px=24,
                    color_bgr=color,
                )

            draw_text_utf8(
                frame,
                "N — שמירה/שם חדש בטרמינל   |   q — יציאה",
                (8, frame.shape[0] - 16),
                font_px=18,
                color_bgr=(200, 200, 200),
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("n"), ord("N")):
                _handle_reenroll_key()
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def _try_reenroll_from_terminal(
    webcam,
    store: FaceProfileStore,
    *,
    embedding_backend: str,
    enroll_frames: int,
    enroll_window_name: str,
    face_lm: FaceLandmarkerDetector | None,
    insight,
) -> None:
    print("\n[identity] הקלד שם ולחץ Enter (שורה ריקה = ביטול)")
    try:
        name = input("שם: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("[identity] בוטל.")
        return
    if not name:
        print("[identity] בלי שם — לא נשמר.")
        return
    eb = str(embedding_backend).lower()
    try:
        if eb == "insightface" and insight is not None:
            run_face_enrollment_insightface(
                webcam,
                insight,
                store,
                enroll_frames=enroll_frames,
                display_name=name,
                window_name=enroll_window_name,
            )
        elif face_lm is not None:
            run_face_enrollment(
                webcam,
                face_lm,
                store,
                enroll_frames=enroll_frames,
                display_name=name,
                window_name=enroll_window_name,
            )
        else:
            print("[identity] אין מודל מותאם לרישום.")
            return
        store.display_name = name
        print(f"[identity] נשמר פרופיל עבור «{name}». ממשיכים בתצוגה.")
    except RuntimeError as exc:
        print(f"[identity] רישום לא הושלם: {exc}")
