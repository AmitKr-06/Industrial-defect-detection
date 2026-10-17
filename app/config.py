"""
Configuration for the FastAPI service.

Reads environment variables so the same image works both locally (loading
artifacts from disk) and in production (loading the champion from the MLflow
registry). All defaults assume the local, dev-friendly setup.
"""

import os
from pathlib import Path

# ----------------------------- Paths -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Where the API writes generated heatmaps. Cleaned up periodically.
UPLOAD_DIR = PROJECT_ROOT / "uploads" / "heatmaps"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Model source -----------------------------
# "local"    -> load models/memory_bank.pt + models/threshold.json
# "registry" -> load models:/<MODEL_NAME>@<MODEL_ALIAS> via MLflow
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local").lower()

# Registry-only settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")   # required if MODEL_SOURCE=registry
MODEL_NAME = os.getenv("MODEL_NAME", "patchcore-metal-nut")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# ----------------------------- Server -----------------------------
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))

# ----------------------------- Uploads -----------------------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp"}

# Heatmaps older than this are deleted (in seconds). 0 disables cleanup.
HEATMAP_TTL_SECONDS = int(os.getenv("HEATMAP_TTL_SECONDS", "3600"))