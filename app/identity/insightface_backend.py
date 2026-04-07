"""ArcFace embeddings via InsightFace (buffalo_l)."""
from __future__ import annotations

import cv2
import numpy as np


class InsightFaceBackend:
    def __init__(self, config: dict) -> None:
        fcfg = config.get("fpv_faces", {})
        det = int(fcfg.get("insightface_det_size", 640))
        self._det_size = (det, det)
        self._ctx_id = int(fcfg.get("insightface_ctx_id", -1))
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace לא מותקן. הרץ: pip install insightface onnxruntime"
            ) from exc

        self._app = FaceAnalysis(name="buffalo_l")
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)

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

    def draw_bbox(self, frame: np.ndarray, bbox: np.ndarray | None) -> None:
        if bbox is None or len(bbox) < 4:
            return
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def close(self) -> None:
        pass
