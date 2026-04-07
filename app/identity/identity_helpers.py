"""Shared setup for identity (landmarks vs InsightFace ArcFace)."""
from __future__ import annotations

from typing import Any


def resolve_identity_detectors(
    config: dict,
) -> tuple[str, float, Any | None, Any | None]:
    """
    Returns (backend_name, threshold, face_landmarker_or_none, insightface_or_none).

    Default backend is insightface (ArcFace) — much more discriminative than landmarks.
    Falls back to landmarks if InsightFace is not installed or fails to load.
    """
    icfg = config.get("identity", {}) or {}
    fcfg = config.get("fpv_faces", {}) or {}
    eb = str(icfg.get("embedding_backend", "insightface")).lower()
    th_land = float(icfg.get("match_threshold", 0.88))
    th_if = float(
        icfg.get(
            "match_threshold_insightface",
            fcfg.get("match_threshold_insightface", 0.42),
        )
    )

    face_lm: Any | None = None
    insight: Any | None = None

    if eb == "insightface":
        try:
            from app.identity.insightface_backend import InsightFaceBackend

            insight = InsightFaceBackend(config)
        except Exception as exc:
            print(
                f"[identity] InsightFace לא מוכן ({exc}) — "
                "משתמשים ב-MediaPipe landmarks (פחות מדויק לזיהוי אנשים שונים)."
            )
            eb = "landmarks"

    if eb == "landmarks":
        from app.detection.face_landmarker_detector import FaceLandmarkerDetector

        face_lm = FaceLandmarkerDetector()

    threshold = th_if if eb == "insightface" else th_land
    return eb, threshold, face_lm, insight


def warn_profile_backend_mismatch(store, requested_backend: str) -> None:
    """If an existing profile was saved with a different backend, warn."""
    if not store.exists():
        return
    loaded = store.load()
    if loaded is None:
        return
    _emb, _name, prof_backend = loaded
    if prof_backend != requested_backend:
        print(
            f"[identity] זהירות: פרופיל קיים נשמר עם "
            f"'{prof_backend}' אבל המצב הוא '{requested_backend}'. "
            "מחק או החלף את profile.npz, או לחץ N במצב identity ורשום מחדש."
        )
