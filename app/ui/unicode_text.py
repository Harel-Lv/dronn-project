"""Draw UTF-8 text (e.g. Hebrew) on OpenCV BGR frames using Pillow."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def _truetype_font(size: int):
    from PIL import ImageFont

    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates: list[Path] = [
        Path(windir) / "Fonts" / "seguiui.ttf",
        Path(windir) / "Fonts" / "segui.ttf",
        Path(windir) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for p in candidates:
        try:
            if p.is_file():
                return ImageFont.truetype(str(p), size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text_utf8(
    frame_bgr: np.ndarray,
    text: str,
    position_xy: tuple[int, int],
    *,
    font_px: int = 22,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    In-place draw. Falls back to ASCII-stripped cv2.putText if Pillow is missing.
    """
    if not text:
        return
    x, y = int(position_xy[0]), int(position_xy[1])
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        safe = text.encode("ascii", "replace").decode("ascii") or "?"
        cv2.putText(
            frame_bgr,
            safe,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        return

    font = _truetype_font(font_px)
    rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # Pillow draws from top-left of glyph box; y is top of text line
    draw.text((x, y), text, font=font, fill=rgb)
    out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    frame_bgr[:] = out
