"""Multiple enrolled faces for FPV / gallery matching."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from app.identity.face_profile import cosine_match


def _safe_filename_stem(display_name: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\-]+", "_", display_name.strip(), flags=re.UNICODE)
    s = s.strip("_")[:max_len] or "person"
    return s


def save_embedding_to_gallery_file(
    gallery_dir: str | Path,
    embedding: np.ndarray,
    display_name: str,
    *,
    embedding_backend: str = "landmarks",
) -> Path | None:
    name = str(display_name).strip()
    if not name:
        return None
    d = Path(gallery_dir)
    d.mkdir(parents=True, exist_ok=True)
    emb = np.asarray(embedding, dtype=np.float64).ravel()
    n = float(np.linalg.norm(emb))
    if n < 1e-9:
        return None
    emb = emb / n
    stem = _safe_filename_stem(name)
    path = d / f"{stem}.npz"
    if path.is_file():
        path = d / f"{stem}_{int(time.time())}.npz"
    try:
        np.savez_compressed(
            path,
            embedding=emb.astype(np.float64),
            name=name,
            embedding_backend=str(embedding_backend),
        )
        return path
    except Exception:
        return None


def prompt_display_name_mandatory() -> str:
    while True:
        try:
            raw = input("שם למאגר (חובה, יוצג בזיהוי): ").strip()
        except EOFError:
            return ""
        if raw:
            return raw
        print("  חובה להזין שם — נסה שוב.")


def refresh_gallery_with_runtime_allowed(
    config: dict,
    runtime_allowed_names: list[str],
) -> "FaceGallery":
    g = face_gallery_from_config(config)
    for n in runtime_allowed_names:
        g.allow_additional_name(n)
    return g


class FaceGallery:
    def __init__(
        self,
        gallery_dir: str | Path,
        match_threshold: float = 0.88,
        *,
        allowed_names: list[str] | None = None,
        min_best_vs_second_gap: float = 0.0,
    ) -> None:
        self.gallery_dir = Path(gallery_dir)
        self.match_threshold = float(match_threshold)
        self._min_best_vs_second_gap = float(min_best_vs_second_gap)
        self._allowed_normalized: set[str] | None = None
        if allowed_names:
            self._allowed_normalized = {
                str(n).strip().casefold()
                for n in allowed_names
                if str(n).strip()
            }
        self._profiles: list[tuple[np.ndarray, str]] = []
        self.reload()

    def allow_additional_name(self, name: str) -> None:
        n = str(name).strip()
        if not n or self._allowed_normalized is None:
            return
        self._allowed_normalized.add(n.casefold())

    def reload(self) -> int:
        self._profiles.clear()
        if not self.gallery_dir.is_dir():
            return 0
        for p in sorted(self.gallery_dir.glob("*.npz")):
            try:
                data = np.load(p, allow_pickle=True)
                if "embedding" not in data.files:
                    continue
                emb = np.asarray(data["embedding"], dtype=np.float64).ravel()
                n = float(np.linalg.norm(emb))
                if n < 1e-9:
                    continue
                emb = emb / n
                name = str(data["name"]) if "name" in data.files else p.stem
                self._profiles.append((emb, name))
            except Exception:
                continue
        return len(self._profiles)

    def add_from_npz_file(
        self,
        path: str | Path,
        *,
        display_name: str | None = None,
    ) -> bool:
        p = Path(path)
        if not p.is_file():
            return False
        try:
            data = np.load(p, allow_pickle=True)
            if "embedding" not in data.files:
                return False
            emb = np.asarray(data["embedding"], dtype=np.float64).ravel()
            n = float(np.linalg.norm(emb))
            if n < 1e-9:
                return False
            emb = emb / n
            name = display_name or (
                str(data["name"]) if "name" in data.files else p.stem
            )
            self._profiles.append((emb, str(name)))
            return True
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self._profiles)

    def _compatible_profiles(self, emb: np.ndarray) -> list[tuple[np.ndarray, str]]:
        d = int(np.asarray(emb).size)
        return [(r, n) for r, n in self._profiles if int(np.asarray(r).size) == d]

    def closest_profile_score(
        self, embedding: np.ndarray | None
    ) -> tuple[str | None, float]:
        if embedding is None or not self._profiles:
            return None, 0.0
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        n = float(np.linalg.norm(emb))
        if n < 1e-9:
            return None, 0.0
        emb = emb / n
        best_name: str | None = None
        best_score = -1.0
        for ref, name in self._compatible_profiles(emb):
            s = cosine_match(ref, emb)
            if s > best_score:
                best_score = s
                best_name = name
        return best_name, best_score

    def has_profile_name(self, display_name: str) -> bool:
        want = str(display_name).strip().casefold()
        if not want:
            return False
        for _, prof_name in self._profiles:
            if str(prof_name).strip().casefold() == want:
                return True
        return False

    def best_match(self, embedding: np.ndarray | None) -> tuple[str | None, float]:
        if embedding is None or not self._profiles:
            return None, 0.0
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        n = float(np.linalg.norm(emb))
        if n < 1e-9:
            return None, 0.0
        emb = emb / n
        per_name: dict[str, tuple[float, str]] = {}
        for ref, name in self._compatible_profiles(emb):
            s = cosine_match(ref, emb)
            key = str(name).strip().casefold()
            if not key:
                continue
            display = str(name).strip()
            if key not in per_name or s > per_name[key][0]:
                per_name[key] = (s, display)
        if not per_name:
            return None, 0.0
        ranked = sorted(per_name.values(), key=lambda x: x[0], reverse=True)
        best_score, best_name = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else None

        if (
            self._min_best_vs_second_gap > 0
            and second_score is not None
            and (best_score - second_score) < self._min_best_vs_second_gap
        ):
            return None, best_score

        if best_score >= self.match_threshold:
            if self._allowed_normalized is not None:
                if best_name.strip().casefold() not in self._allowed_normalized:
                    return None, best_score
            return best_name, best_score
        return None, best_score


def face_gallery_from_config(config: dict) -> FaceGallery:
    fcfg = config.get("fpv_faces", {})
    gdir = str(fcfg.get("gallery_dir", "data/faces_gallery"))
    backend = str(fcfg.get("embedding_backend", "landmarks")).lower()
    if backend == "insightface":
        thr = float(fcfg.get("match_threshold_insightface", 0.42))
    else:
        thr = float(fcfg.get("match_threshold", 0.93))
    raw_allowed = fcfg.get("allowed_names")
    allowed: list[str] | None = None
    if isinstance(raw_allowed, list) and len(raw_allowed) > 0:
        allowed = [str(x) for x in raw_allowed]
    gap_raw = fcfg.get("min_best_vs_second_gap", 0)
    gap = float(gap_raw) if gap_raw is not None else 0.0
    gallery = FaceGallery(
        gdir,
        match_threshold=thr,
        allowed_names=allowed,
        min_best_vs_second_gap=gap,
    )
    icfg = config.get("identity", {})
    if bool(fcfg.get("include_identity_profile", True)):
        if backend == "insightface":
            print(
                "[fpv_faces] לא טוען profile.npz של identity — לא תואם ל-insightface."
            )
        else:
            id_path = Path(str(icfg.get("profile_path", "data/identity/profile.npz")))
            if id_path.is_file():
                dn = str(icfg.get("display_name", "Owner"))
                gallery.add_from_npz_file(id_path, display_name=dn)
    return gallery


def registration_face_precheck_message(
    gallery: FaceGallery,
    embedding: np.ndarray,
    config: dict,
) -> str | None:
    if len(gallery) == 0:
        return None
    fcfg = config.get("fpv_faces", {})
    raw = fcfg.get("duplicate_registration_threshold")
    if raw is None:
        be = str(fcfg.get("embedding_backend", "landmarks")).lower()
        if be == "insightface":
            raw = fcfg.get("match_threshold_insightface", 0.42)
        else:
            raw = fcfg.get("match_threshold", 0.93)
    thr = float(raw)
    nm, sc = gallery.closest_profile_score(embedding)
    if nm is not None and sc >= thr:
        return (
            f'[מאגר] פרצוף זה כבר רשום כמוכר תחת "{nm}" '
            f"(דמיון {sc:.2f} ≥ {thr:.2f}) — אי אפשר לרשום פעמיים."
        )
    return None


def registration_name_precheck_message(
    gallery: FaceGallery, display_name: str
) -> str | None:
    name = str(display_name).strip()
    if not name:
        return "[מאגר] שם ריק."
    if gallery.has_profile_name(name):
        return f'[מאגר] השם "{name}" כבר קיים במאגר — בחר שם אחר.'
    return None
