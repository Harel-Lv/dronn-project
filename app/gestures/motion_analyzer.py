"""Map MediaPipe hand landmarks to coarse gesture labels."""
from __future__ import annotations

import math

# MediaPipe hand indices
_WRIST = 0
_THUMB_TIP, _THUMB_IP, _THUMB_MCP = 4, 3, 2
_INDEX_TIP, _INDEX_PIP = 8, 6
_MIDDLE_TIP, _MIDDLE_PIP = 12, 10
_MIDDLE_MCP = 9
_RING_TIP, _RING_PIP = 16, 14
_PINKY_TIP, _PINKY_PIP = 20, 18

_G_UNKNOWN = "UNKNOWN"
_G_NONE = "NONE"
_G_FIST = "FIST"
_G_OPEN = "OPEN_PALM"
_G_POINT = "POINT"
_G_PEACE = "PEACE"
_G_THUMB_UP = "THUMB_UP"


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def palm_span_pixels(lm: dict[int, tuple[int, int]]) -> float | None:
    """Scale proxy: מרחק שורש כף–אמצע (MCP) — לכיול מרחק מהמצלמה."""
    if _WRIST not in lm or _MIDDLE_MCP not in lm:
        return None
    return _dist(lm[_WRIST], lm[_MIDDLE_MCP])


def likely_open_palm_for_calibration(
    lm: dict[int, tuple[int, int]],
    *,
    loose_finger_ratio: float = 1.05,
) -> bool:
    """בדיקה רופפת לכיול בלבד — ארבע אצבעות פתוחות (בלי לחייב אגודל)."""
    if _WRIST not in lm:
        return False

    def ext(tip: int, pip: int) -> bool:
        if tip not in lm or pip not in lm:
            return False
        w, t, p = lm[_WRIST], lm[tip], lm[pip]
        return _dist(w, t) > _dist(w, p) * loose_finger_ratio

    idx = ext(_INDEX_TIP, _INDEX_PIP)
    mid = ext(_MIDDLE_TIP, _MIDDLE_PIP)
    ring = ext(_RING_TIP, _RING_PIP)
    pink = ext(_PINKY_TIP, _PINKY_PIP)
    n = sum(1 for x in (idx, mid, ring, pink) if x)
    return n >= 4


class MotionAnalyzer:
    def __init__(
        self,
        use_pose: bool = False,
        *,
        reference_palm_span: float | None = None,
        finger_base_ratio: float = 1.12,
        thumb_base_ratio: float = 1.08,
        palm_scale_clamp: tuple[float, float] = (0.72, 1.48),
    ) -> None:
        self.use_pose = bool(use_pose)
        self.reference_palm_span = reference_palm_span
        self.finger_base_ratio = float(finger_base_ratio)
        self.thumb_base_ratio = float(thumb_base_ratio)
        self._palm_lo, self._palm_hi = palm_scale_clamp

    def set_reference_palm_span(self, ref: float | None) -> None:
        if ref is None or ref < 1e-3:
            self.reference_palm_span = None
        else:
            self.reference_palm_span = float(ref)

    def _finger_ratio(self, lm: dict[int, tuple[int, int]]) -> float:
        palm = palm_span_pixels(lm)
        if palm is None or self.reference_palm_span is None:
            return self.finger_base_ratio
        scale = palm / max(self.reference_palm_span, 1e-6)
        scale = max(self._palm_lo, min(self._palm_hi, scale))
        return self.finger_base_ratio * scale

    def _thumb_ratio(self, lm: dict[int, tuple[int, int]]) -> float:
        palm = palm_span_pixels(lm)
        if palm is None or self.reference_palm_span is None:
            return self.thumb_base_ratio
        scale = palm / max(self.reference_palm_span, 1e-6)
        scale = max(self._palm_lo, min(self._palm_hi, scale))
        return self.thumb_base_ratio * scale

    def _finger_extended(
        self, lm: dict[int, tuple[int, int]], tip: int, pip: int
    ) -> bool:
        if _WRIST not in lm or tip not in lm or pip not in lm:
            return False
        w, t, p = lm[_WRIST], lm[tip], lm[pip]
        ratio = self._finger_ratio(lm)
        return _dist(w, t) > _dist(w, p) * ratio

    def _thumb_extended(self, lm: dict[int, tuple[int, int]]) -> bool:
        if _WRIST not in lm or _THUMB_TIP not in lm or _THUMB_IP not in lm:
            return False
        w, t, ip = lm[_WRIST], lm[_THUMB_TIP], lm[_THUMB_IP]
        ratio = self._thumb_ratio(lm)
        return _dist(w, t) > _dist(w, ip) * ratio

    def analyze_hands(self, hands: list[dict[int, tuple[int, int]]]) -> str:
        if not hands:
            return _G_NONE
        lm = hands[0]
        thumb = self._thumb_extended(lm)
        idx = self._finger_extended(lm, _INDEX_TIP, _INDEX_PIP)
        mid = self._finger_extended(lm, _MIDDLE_TIP, _MIDDLE_PIP)
        ring = self._finger_extended(lm, _RING_TIP, _RING_PIP)
        pink = self._finger_extended(lm, _PINKY_TIP, _PINKY_PIP)

        fingers = (idx, mid, ring, pink)
        n = sum(1 for x in fingers if x)

        # סדר חשוב: אצבע מצביעה לפני "אגודל למעלה" — כדי שלא יבלבל עם הצבעה למעלה
        if n == 0 and not thumb:
            return _G_FIST
        if n >= 4:
            return _G_OPEN
        if n == 2 and idx and mid and not ring and not pink:
            return _G_PEACE
        if n == 1 and idx:
            return _G_POINT
        # אגודל למעלה קלאסי: רק האגודל בחוץ, ארבע האצבעות מקופות (בלי אצבע מצביעה)
        if thumb and n == 0:
            return _G_THUMB_UP
        return _G_UNKNOWN

    def analyze(
        self,
        hands: list[dict[int, tuple[int, int]]],
        pose_landmarks: dict[int, tuple[int, int]] | None = None,
    ) -> str:
        g = self.analyze_hands(hands)
        if self.use_pose and pose_landmarks and g == _G_NONE:
            if _arms_raised(pose_landmarks):
                return _G_OPEN
        return g


def _arms_raised(lm: dict[int, tuple[int, int]]) -> bool:
    need = (11, 12, 15, 16)
    if not all(k in lm for k in need):
        return False
    ls, rs, lw, rw = lm[11], lm[12], lm[15], lm[16]
    return lw[1] < ls[1] and rw[1] < rs[1]
