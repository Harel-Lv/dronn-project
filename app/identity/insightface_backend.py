"""ArcFace embeddings via InsightFace (buffalo_l)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# FaceAnalysis takes seconds to load; mode switches reuse one instance per settings.
_APP_CACHE: dict[tuple[int, int], Any] = {}


def _load_face_analysis(det_size: tuple[int, int], ctx_id: int) -> Any:
    key = (det_size[0], ctx_id)
    app = _APP_CACHE.get(key)
    if app is not None:
        return app
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "InsightFace is not installed. Run: pip install insightface onnxruntime"
        ) from exc
    # Only detection + ArcFace; buffalo_l also ships landmark and gender/age models we never use.
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    _APP_CACHE[key] = app
    return app


class InsightFaceBackend:
    def __init__(self, config: dict) -> None:
        icfg = config.get("identity", {}) or {}
        fcfg = config.get("fpv_faces", {}) or {}
        det = int(
            icfg.get("insightface_det_size", fcfg.get("insightface_det_size", 320))
        )
        self._det_size = (det, det)
        self._ctx_id = int(fcfg.get("insightface_ctx_id", -1))
        self._app = _load_face_analysis(self._det_size, self._ctx_id)

    def embed(self, bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        faces = self._app.get(bgr)
        if not faces:
            return None, None

        def _area(face) -> float:
            b = np.asarray(face.bbox, dtype=np.float64)
            if len(b) < 4:
                return 0.0
            return float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))

        f = max(faces, key=_area)
        e = np.asarray(f.embedding, dtype=np.float64).ravel()
        n = float(np.linalg.norm(e))
        if n < 1e-9:
            return None, None
        e = e / n
        bbox = np.asarray(f.bbox, dtype=np.int32)
        return e, bbox

    def detect_largest_bbox(self, bgr: np.ndarray) -> np.ndarray | None:
        """Largest face box only — skips ArcFace (for modes that don't need identity)."""
        bboxes, _kpss = self._app.det_model.detect(bgr, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]).clip(min=0) * (bboxes[:, 3] - bboxes[:, 1]).clip(min=0)
        return np.asarray(bboxes[int(np.argmax(areas)), :4], dtype=np.int32)

    def draw_bbox(self, frame: np.ndarray, bbox: np.ndarray | None) -> None:
        if bbox is None or len(bbox) < 4:
            return
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def close(self) -> None:
        # The model is shared across mode switches (see _APP_CACHE).
        pass
