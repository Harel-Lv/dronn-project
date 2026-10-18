"""Face identity: cosine match on stored embedding (landmarks or ArcFace)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _unpack_npz_str(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    raw = np.asarray(data[key])
    if raw.dtype == object:
        return str(raw.flat[0])
    if raw.ndim == 0:
        return str(raw.item())
    return str(raw.flat[0])


# Outer eye corners in the MediaPipe face mesh; their distance sets the face scale.
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
# Old embeddings were raw (x, y, z) per landmark (478 * 3); the shape-normalized
# embedding is (x, y) only, so profiles saved before the change never match by size.
LEGACY_LANDMARK_EMBEDDING_SIZE = 478 * 3


def face_landmarks_to_embedding(landmarks_list: list) -> np.ndarray | None:
    """Face *shape* vector: independent of where the face is in the frame and how far it is."""
    if not landmarks_list or len(landmarks_list) <= _RIGHT_EYE_OUTER:
        return None
    pts = np.array([(lm.x, lm.y) for lm in landmarks_list], dtype=np.float64)
    pts -= pts.mean(axis=0)
    scale = float(np.linalg.norm(pts[_RIGHT_EYE_OUTER] - pts[_LEFT_EYE_OUTER]))
    if scale < 1e-6:
        return None
    arr = (pts / scale).ravel()
    n = np.linalg.norm(arr)
    if n < 1e-9:
        return None
    return arr / n


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray | None:
    if not embeddings:
        return None
    stacked = np.stack(embeddings, axis=0)
    mean = np.mean(stacked, axis=0)
    n = np.linalg.norm(mean)
    if n < 1e-9:
        return None
    return mean / n


def cosine_match(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class FaceProfileStore:
    def __init__(self, profile_path: str, display_name: str = "Owner"):
        self.path = Path(profile_path)
        self.display_name = display_name
        # (mtime_ns, size, parsed profile); verify() runs every few frames.
        self._cache: tuple[int, int, tuple[np.ndarray, str, str] | None] | None = None

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> tuple[np.ndarray, str, str] | None:
        """Returns (normalized embedding, display name, backend id: landmarks | insightface)."""
        try:
            st = self.path.stat()
        except OSError:
            self._cache = None
            return None
        if self._cache is not None and self._cache[:2] == (st.st_mtime_ns, st.st_size):
            return self._cache[2]
        try:
            with np.load(self.path, allow_pickle=True) as data:
                if "embedding" not in data.files:
                    return None
                emb = np.asarray(data["embedding"], dtype=np.float64).ravel()
                name = _unpack_npz_str(data, "name", self.display_name)
                backend = _unpack_npz_str(data, "backend", "landmarks")
        except Exception:
            # File may be mid-write by enrollment: keep using the last good profile.
            return self._cache[2] if self._cache is not None else None
        n = float(np.linalg.norm(emb))
        if n < 1e-9:
            return None
        emb = emb / n
        if backend not in ("landmarks", "insightface"):
            backend = "landmarks"
        parsed = (emb, name, backend)
        self._cache = (st.st_mtime_ns, st.st_size, parsed)
        return parsed

    def save(
        self,
        embedding: np.ndarray,
        name: str | None = None,
        *,
        backend: str = "landmarks",
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = None
        np.savez_compressed(
            self.path,
            embedding=embedding.astype(np.float64),
            name=np.str_(name or self.display_name),
            backend=np.str_(backend),
        )

    def verify(self, embedding: np.ndarray | None, threshold: float) -> tuple[bool, float, str]:
        loaded = self.load()
        if loaded is None or embedding is None:
            return False, 0.0, self.display_name
        ref, stored_name, _backend = loaded
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        ne = float(np.linalg.norm(emb))
        if ne < 1e-9:
            return False, 0.0, stored_name
        emb = emb / ne
        if ref.shape != emb.shape:
            return False, 0.0, stored_name
        score = cosine_match(ref, emb)
        return score >= threshold, score, stored_name
