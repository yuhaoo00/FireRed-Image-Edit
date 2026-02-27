"""Configuration for FireRed Agent."""

import os

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Image stitching defaults
# ---------------------------------------------------------------------------
# Target total area (pixels) for the stitched output canvas
STITCH_TARGET_AREA: int = 1024 * 1024  # ~1 mega-pixel
# Max number of output images that FireRed-Image-Edit accepts
MAX_OUTPUT_IMAGES: int = 3
# Min / max aspect ratio for the stitched canvas
STITCH_MIN_ASPECT: float = 0.5   # portrait  (H = 2W)
STITCH_MAX_ASPECT: float = 2.0   # landscape (W = 2H)
# Padding colour when a small gap remains
STITCH_PAD_COLOR: tuple[int, int, int] = (255, 255, 255)

# ---------------------------------------------------------------------------
# Recaption
# ---------------------------------------------------------------------------
RECAPTION_TARGET_LENGTH: int = 512  # target word/character count
