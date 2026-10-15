"""Alternative PGM presentation, with no target-dependent image processing."""
from __future__ import annotations

from PIL import Image


def split_composite(image: Image.Image) -> list[Image.Image]:
    """Losslessly recover the native 16 panels from matrix-3x3-options-2x4-v1.

    This only accepts the canonical renderer's dimensions. It neither resizes
    panels nor includes rendered answer labels or the missing-cell placeholder.
    """
    if image.size != (700, 1016):
        raise ValueError(f"Unexpected PGM composite dimensions: {image.size}")
    image = image.convert("RGB")
    positions = [(98 + (i % 3) * 172, 42 + (i // 3) * 172) for i in range(8)]
    positions += [(12 + (i % 4) * 172, 630 + (i // 4) * 202) for i in range(8)]
    return [image.crop((x, y, x + 160, y + 160)) for x, y in positions]


def messages(image: Image.Image, prompt: str, presentation: str) -> list[dict]:
    if presentation == "composite":
        content = [{"type": "image", "image": image.convert("RGB")}]
    elif presentation == "separate":
        content = []
        for i, panel in enumerate(split_composite(image)):
            label = (f"Context panel: row {i // 3 + 1}, column {i % 3 + 1}."
                     if i < 8 else f"Answer option {chr(65 + i - 8)}.")
            content.extend([{"type": "text", "text": label}, {"type": "image", "image": panel}])
    else:
        raise ValueError(f"Unknown presentation: {presentation}")
    # Keep the established task instruction verbatim, after the images.
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
