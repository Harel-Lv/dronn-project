"""Stabilize high-level intents — stricter confirmation for TAKEOFF / LAND."""
from __future__ import annotations

from app.gestures.gesture_rules import LAND, TAKEOFF

_CRITICAL = frozenset({TAKEOFF, LAND})


class GestureIntentStabilizer:
    """
    Like TrackingCommandStabilizer, but uses a higher frame count when the
    *candidate* intent is TAKEOFF or LAND; other intents use a lower count.
    """

    def __init__(
        self,
        *,
        stable_frames_normal: int = 4,
        stable_frames_critical: int = 8,
        initial_intent: str = "HOVER",
    ) -> None:
        self.stable_frames_normal = max(1, int(stable_frames_normal))
        self.stable_frames_critical = max(1, int(stable_frames_critical))
        self.displayed_intent = initial_intent
        self._candidate: str | None = None
        self._candidate_count = 0

    def _required_for(self, intent: str) -> int:
        return (
            self.stable_frames_critical
            if intent in _CRITICAL
            else self.stable_frames_normal
        )

    def update(self, intent: str) -> str:
        if intent == self.displayed_intent:
            self._candidate = None
            self._candidate_count = 0
            return self.displayed_intent

        if intent == self._candidate:
            self._candidate_count += 1
            need = self._required_for(intent)
            if self._candidate_count >= need:
                self.displayed_intent = intent
                self._candidate = None
                self._candidate_count = 0
            return self.displayed_intent

        self._candidate = intent
        self._candidate_count = 1
        return self.displayed_intent

    def reset_to(self, intent: str) -> None:
        """Force displayed intent (e.g. after emergency land)."""
        self.displayed_intent = intent
        self._candidate = None
        self._candidate_count = 0
