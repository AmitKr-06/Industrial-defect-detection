"""
Filesystem-backed store for generated heatmap PNGs.

Heatmaps are written to uploads/heatmaps/<uuid>.png and served via /heatmap/<uuid>.
A simple TTL cleanup removes files older than HEATMAP_TTL_SECONDS.
"""

import time
import uuid
from pathlib import Path

from PIL import Image

from app import config


def save_heatmap(pil_image: Image.Image) -> str:
    """Save a PIL image to disk and return a UUID that identifies it."""
    file_id = uuid.uuid4().hex
    path = config.UPLOAD_DIR / f"{file_id}.png"
    pil_image.save(path, "PNG")
    return file_id


def heatmap_path(file_id: str) -> Path | None:
    """Return the path to a heatmap if it exists and hasn't expired."""
    path = config.UPLOAD_DIR / f"{file_id}.png"
    if not path.exists():
        return None
    if config.HEATMAP_TTL_SECONDS > 0:
        age = time.time() - path.stat().st_mtime
        if age > config.HEATMAP_TTL_SECONDS:
            path.unlink(missing_ok=True)
            return None
    return path


def cleanup_expired() -> int:
    """Delete all expired heatmaps. Returns the count deleted."""
    if config.HEATMAP_TTL_SECONDS <= 0:
        return 0
    now = time.time()
    removed = 0
    for p in config.UPLOAD_DIR.glob("*.png"):
        if now - p.stat().st_mtime > config.HEATMAP_TTL_SECONDS:
            p.unlink(missing_ok=True)
            removed += 1
    return removed