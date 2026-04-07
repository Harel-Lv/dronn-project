"""Pick one hand for gesture control using MediaPipe handedness."""
from __future__ import annotations


def select_dominant_hand_landmarks(
    hands_with_labels: list[tuple[dict, str]],
    dominant: str,
) -> dict | None:
    """
    :param hands_with_labels: (landmark_dict, label) per detected hand;
        label is usually 'Left' or 'Right' from MediaPipe.
    :param dominant: 'left' | 'right' | 'any' (case-insensitive).
    """
    if not hands_with_labels:
        return None
    d = (dominant or "any").strip().lower()
    if d in ("any", "", "both"):
        return hands_with_labels[0][0]

    want = "Right" if d == "right" else "Left" if d == "left" else None
    if want is None:
        return hands_with_labels[0][0]

    for lm, label in hands_with_labels:
        if (label or "").strip() == want:
            return lm
    # חשוב: לא נשבור התנהגות — אם אין התאמה (מצלמה / חוסר ביטחון), ניקח יד ראשונה
    return hands_with_labels[0][0]
