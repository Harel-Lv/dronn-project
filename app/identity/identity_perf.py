"""Frame skipping and resize helpers for InsightFace identity loops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def identity_perf_settings(config: dict) -> tuple[int, int]:
    """Return (process_every_n_frames, max_width) from identity / fpv_faces config."""
    icfg = config.get("identity", {}) or {}
    fcfg = config.get("fpv_faces", {}) or {}
    every = max(
        1,
        int(
            icfg.get(
                "insightface_process_every_n_frames",
                fcfg.get("insightface_process_every_n_frames", 3),
            )
        ),
    )
    max_w = max(
        160,
        int(
            icfg.get(
                "insightface_max_width",
                fcfg.get("insightface_max_width", 480),
            )
        ),
    )
    return every, max_w


def resize_frame_max_width(
    frame: np.ndarray, max_width: int
) -> tuple[np.ndarray, float]:
    """Return (possibly smaller frame, multiplier to map coords back to full frame)."""
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, 1.0 / scale


def scale_bbox(
    bbox: np.ndarray | None, inv_scale: float
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    b = np.asarray(bbox, dtype=np.float64)
    if len(b) < 4:
        return None
    scaled = (b[:4] * inv_scale).astype(np.int32)
    return int(scaled[0]), int(scaled[1]), int(scaled[2]), int(scaled[3])


@dataclass
class IdentityInsightResult:
    embedding: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    ok: bool
    score: float
    name: str
    face_detected: bool


class CachedInsightFaceVerifier:
    """Run InsightFace every N frames; reuse last result between runs."""

    def __init__(
        self,
        insight: Any,
        store: Any,
        *,
        match_threshold: float,
        process_every_n_frames: int = 3,
        max_width: int = 480,
    ) -> None:
        self._insight = insight
        self._store = store
        self._threshold = float(match_threshold)
        self._every = max(1, int(process_every_n_frames))
        self._max_width = max(160, int(max_width))
        self._frame_idx = 0
        self._cached = IdentityInsightResult(
            None, None, False, 0.0, str(getattr(store, "display_name", "")), False
        )

    def process(self, frame: np.ndarray) -> IdentityInsightResult:
        run_detection = self._frame_idx % self._every == 0
        self._frame_idx += 1

        if not run_detection:
            return self._cached

        small, inv_scale = resize_frame_max_width(frame, self._max_width)
        emb, bbox = self._insight.embed(small)
        if bbox is None or emb is None:
            self._cached = IdentityInsightResult(
                None,
                None,
                False,
                0.0,
                str(self._store.display_name),
                False,
            )
            return self._cached

        scaled_bbox = scale_bbox(bbox, inv_scale)
        ok, score, name = self._store.verify(emb, self._threshold)
        self._cached = IdentityInsightResult(
            emb, scaled_bbox, ok, score, name, True
        )
        return self._cached

    def reset(self) -> None:
        self._frame_idx = 0
        self._cached = IdentityInsightResult(
            None, None, False, 0.0, str(self._store.display_name), False
        )
